"""
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Shuvi

Item2Vec (SGNS) training kernel: learn 64-dim embeddings + evaluation.

Trains Skip-Gram with Negative Sampling on user listening sequences.
Input:  events.parquet + vocab.parquet from lb-trainprep
Output: models/item2vec_final.npy, reports/eval_report.md, metrics.json

Pipeline:
  Step 1 — Load data into CSR (per-user sequences, two vectorized passes)
  Step 2 — Pair generation (dynamic window W=5, subsampling t=1e-3, vectorized)
  Step 3 — SGNS training (PyTorch, GPU if available)
  Step 4 — Evaluation (Recall@20/100, MRR@10, HitRate@20) + baselines
  Step 5 — README.md

Design notes:
  - CSR layout: one int32 array + offsets. NO per-user Python lists
    (1.27B Python ints would be ~45GB RAM — Kaggle has 13GB).
  - Pair generation vectorized per user: numpy masks per window distance d.
  - Negatives are uniform (NOT popularity-biased) for v1; may sample the
    positive itself (classic SGNS ignores this, harmless).
  - Subsampling: keep_prob = min(1, sqrt(t/freq) + t/freq). NOTE: with
    t=1e-3 and a 1.25B-event corpus, only items with >~2M listens are
    affected — effectively a no-op on this data (logged, not hidden).
  - Eval: context = mean of last 50 train embeddings; repeats NOT excluded.
  - Checkpoints: npy not parquet (2.8M x 64 too large for parquet).
"""
import json, os, sys, time
from collections import defaultdict
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────
EMB_DIM         = 64
WINDOW          = 5             # max skip-gram window
SUBSAMPLE_T     = 1e-3          # subsampling threshold (see note above)
NEG_SAMPLES     = 5             # negatives per positive pair
BATCH_PAIRS     = 1_000_000     # pairs per training buffer flush
CHUNK           = 100_000       # rows per optimizer step within a batch
EPOCHS          = 2
LR              = 5e-4
SEED            = 42
LOG_INTERVAL    = 100           # log every N optimizer batches
EVAL_SAMPLE     = 5_000         # users to sample for eval
EVAL_CTX_LEN    = 50            # last N train events for context
RECALL_KS       = [20, 100]
MRR_K           = 10
HITRATE_K       = 20

# ── EXPECTED VALUES (embedded for sanity checking) ─────────────────────
EXPECTED = dict(
    n_users=36970,
    n_items=2803656,
    train_rows_upper=1_270_000_000,
)

import numpy as np
import pyarrow.parquet as pq

T0 = time.time()
def log(m): print(f"[{int(time.time()-T0):>5}s] {m}", flush=True)


def _ensure_torch_gpu():
    """Kaggle's preinstalled torch dropped sm_60 (Tesla P100). If we land on a
    P100, install the cu118 build (last with sm_60 support) BEFORE torch import."""
    import subprocess
    try:
        cap = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return  # no nvidia-smi -> CPU-only environment; preinstalled torch fine
    if cap.startswith("6."):
        log(f"GPU compute cap {cap} (P100) unsupported by preinstalled torch; installing cu118 build")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "torch==2.4.1+cu118",
                        "--index-url", "https://download.pytorch.org/whl/cu118"],
                       check=True)


_ensure_torch_gpu()


# ═══════════════════════════════════════════════════════════════════════
# STEP 1 — Load data into CSR (two vectorized passes, no Python lists)
# ═══════════════════════════════════════════════════════════════════════

def load_csr(events_path, n_items):
    """Load train split into CSR arrays; collect test item sets in pass 1.

    Returns:
      offsets:     int64 (n_users+1,) — user's train events are
                   data[offsets[i]:offsets[i+1]], file order (= ts order)
      data:        int32 concatenated train item_ids
      user_ids:    int32 sorted unique user ids
      user_index:  dict user_id -> CSR row index
      train_counts:int64 (n_items,) per-item train frequency (padded)
      user_test:   dict user_id -> set(test item_ids)
    """
    log(f"Loading events from {events_path}")
    pf = pq.ParquetFile(str(events_path))

    # ── Pass 1: per-user train counts + test sets + order check ──
    user_train_counts = defaultdict(int)   # 37K entries
    user_test = defaultdict(set)            # ~5M entries total
    total_rows = 0
    user_regressions = 0
    last_user = -1
    batch_num = 0

    for batch in pf.iter_batches(batch_size=2_000_000,
                                  columns=["user", "item_id", "split"]):
        batch_num += 1
        users_np  = batch.column("user").to_numpy()
        items_np  = batch.column("item_id").to_numpy()
        splits_np = batch.column("split").to_numpy()
        total_rows += batch.num_rows

        if len(users_np):
            if last_user >= 0 and int(users_np[0]) < last_user:
                user_regressions += 1
            last_user = int(users_np[-1])

        m = splits_np == 0
        if m.any():
            u, c = np.unique(users_np[m], return_counts=True)
            for uid, cnt in zip(u.tolist(), c.tolist()):
                user_train_counts[uid] += cnt

        m = splits_np == 2
        if m.any():
            u = users_np[m]; it = items_np[m]
            for uid in np.unique(u).tolist():
                user_test[uid].update(it[u == uid].tolist())

        if batch_num % 50 == 0:
            log(f"  p1 batch {batch_num:>5}  rows={total_rows:>13,}  "
                f"users={len(user_train_counts):>7,}")

    log(f"  p1 done: {total_rows:,} rows, {len(user_train_counts):,} users, "
        f"{user_regressions} user regressions")

    # ── Build CSR skeleton ──
    user_ids = np.array(sorted(user_train_counts.keys()), dtype=np.int32)
    user_index = {int(u): i for i, u in enumerate(user_ids.tolist())}
    offsets = np.zeros(len(user_ids) + 1, dtype=np.int64)
    for i, u in enumerate(user_ids.tolist()):
        offsets[i + 1] = offsets[i] + user_train_counts[u]
    data = np.empty(int(offsets[-1]), dtype=np.int32)

    # ── Pass 2: fill data array (file is user-grouped, ts-ordered) ──
    fill = defaultdict(int)                # user_id -> events written
    pf = pq.ParquetFile(str(events_path))
    batch_num = 0
    for batch in pf.iter_batches(batch_size=2_000_000,
                                  columns=["user", "item_id", "split"]):
        batch_num += 1
        users_np  = batch.column("user").to_numpy()
        items_np  = batch.column("item_id").to_numpy()
        splits_np = batch.column("split").to_numpy()
        m = splits_np == 0
        if not m.any():
            continue
        u = users_np[m]; it = items_np[m]
        for uid in np.unique(u).tolist():     # ~58 users per 2M batch
            um = u == uid
            n = int(um.sum())
            start = offsets[user_index[uid]] + fill[uid]
            data[start:start + n] = it[um]
            fill[uid] += n
        if batch_num % 50 == 0:
            log(f"  p2 batch {batch_num:>5}  filled={int(offsets[-1]) and sum(fill.values()):,}")

    train_counts = np.bincount(data, minlength=n_items).astype(np.int64)
    log(f"  CSR: {len(data):,} train events, {len(user_ids):,} users, "
        f"max_item_id={int(data.max()) if len(data) else -1}")
    return (offsets, data, user_ids, user_index, train_counts, user_test)


def load_n_items(vocab_path):
    """Vocab size from vocab.parquet (item_ids are dense 0..N-1 by construction)."""
    vt = pq.read_table(str(vocab_path), columns=["item_id"])
    ids = vt.column("item_id").to_numpy()
    return int(ids.max()) + 1


# ═══════════════════════════════════════════════════════════════════════
# STEP 2 — Pair generation (vectorized per user)
# ═══════════════════════════════════════════════════════════════════════

def compute_keep_probs(train_counts, t=SUBSAMPLE_T):
    """Subsampling keep probabilities per item (word2vec formula)."""
    total = train_counts.sum()
    if total == 0:
        return np.ones(len(train_counts), dtype=np.float32)
    freq = np.maximum(train_counts.astype(np.float64) / total, 1e-12)
    keep = np.minimum(1.0, np.sqrt(t / freq) + t / freq).astype(np.float32)
    keep[train_counts == 0] = 0.0
    return keep


def generate_pairs_for_user(seq, keep_probs, window, rng):
    """Skip-gram pairs for one user's sequence. Vectorized.

    Dynamic window: per center position i, effective window drawn 1..W;
    pairs (i, j) for all j != i with |i-j| <= eff[i].
    Subsampling drops positions (rng.random() > keep) BEFORE windowing.

    Args:
      seq: numpy int32 array (user's train sequence, ts order)
      rng: numpy Generator (deterministic per user/epoch via seed)
    Returns (centers, contexts) int32 arrays (may be empty).
    """
    n = len(seq)
    if n < 2:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    keep_mask = rng.random(n) < keep_probs[seq]
    survivors = np.where(keep_mask)[0]
    if len(survivors) < 2:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    surv = seq[survivors]
    ns = len(surv)
    eff = rng.integers(1, window + 1, size=ns)

    centers, contexts = [], []
    for d in range(1, window + 1):
        if ns <= d:
            break
        # forward pairs: center i, context i+d, needs eff[i] >= d
        m = eff[:-d] >= d
        if m.any():
            centers.append(surv[:-d][m]); contexts.append(surv[d:][m])
        # backward pairs: center i, context i-d, needs eff[i] >= d
        m = eff[d:] >= d
        if m.any():
            centers.append(surv[d:][m]); contexts.append(surv[:-d][m])

    if not centers:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    return np.concatenate(centers), np.concatenate(contexts)


# ═══════════════════════════════════════════════════════════════════════
# STEP 3 — SGNS Training
# ═══════════════════════════════════════════════════════════════════════

def train_sgns(offsets, data, keep_probs, n_items, output_dir,
               epochs=EPOCHS, lr=LR, emb_dim=EMB_DIM, neg=NEG_SAMPLES,
               seed=SEED):
    """Train SGNS embeddings. Returns final center embedding table (numpy)."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Training SGNS: device={device}, {n_items:,} items, dim={emb_dim}")

    center_emb  = (torch.randn(n_items, emb_dim, device=device) * 0.5 / emb_dim)
    context_emb = (torch.randn(n_items, emb_dim, device=device) * 0.5 / emb_dim)
    center_emb.requires_grad_(True)
    context_emb.requires_grad_(True)
    optimizer = torch.optim.Adam([center_emb, context_emb], lr=lr)

    output_dir = Path(output_dir)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)

    n_users = len(offsets) - 1

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        epoch_pairs = 0
        epoch_loss = 0.0
        n_batches = 0

        order = np.arange(n_users)
        np.random.default_rng(seed + epoch).shuffle(order)

        buf_c, buf_ctx = [], []

        def flush():
            nonlocal epoch_pairs, epoch_loss, n_batches
            batch_c = np.concatenate(buf_c)
            batch_ctx = np.concatenate(buf_ctx)
            loss = _train_batch(center_emb, context_emb, optimizer,
                                batch_c, batch_ctx, n_items, neg, device)
            epoch_loss += loss
            n_batches += 1
            epoch_pairs += len(batch_c)
            if n_batches % LOG_INTERVAL == 0:
                el = time.time() - t_epoch
                log(f"  batch {n_batches:>5}  pairs={epoch_pairs:>13,}  "
                    f"loss={epoch_loss / n_batches:.4f}  "
                    f"pairs/s={epoch_pairs / max(el, 0.001):,.0f}")

        for ui in order:
            seq = data[offsets[ui]:offsets[ui + 1]]
            if len(seq) < 2:
                continue
            rng = np.random.default_rng(
                (seed * 1_000_003 + epoch * 100_07 + int(ui)) % (2**31))
            c, ctx = generate_pairs_for_user(seq, keep_probs, WINDOW, rng)
            if len(c):
                buf_c.append(c)
                buf_ctx.append(ctx)
            if sum(len(b) for b in buf_c) >= BATCH_PAIRS:
                flush()
                buf_c, buf_ctx = [], []

        if buf_c:
            flush()

        log(f"  epoch {epoch} done: {epoch_pairs:,} pairs, "
            f"avg_loss={epoch_loss / max(n_batches, 1):.4f}, "
            f"{time.time() - t_epoch:.0f}s")

        np.save(str(output_dir / "models" / f"item2vec_e{epoch}.npy"),
                center_emb.detach().cpu().numpy())

    final = center_emb.detach().cpu().numpy()
    np.save(str(output_dir / "models" / "item2vec_final.npy"), final)
    log(f"  saved models/item2vec_final.npy")
    return final


def _train_batch(center_emb, context_emb, optimizer, batch_c, batch_ctx,
                 n_items, neg, device):
    """One buffer of pairs, trained in CHUNK-sized optimizer steps."""
    import torch
    import torch.nn.functional as F

    c = torch.from_numpy(batch_c).long().to(device)
    ctx = torch.from_numpy(batch_ctx).long().to(device)

    total_loss, n_chunks = 0.0, 0
    for i in range(0, len(c), CHUNK):
        c_chunk, ctx_chunk = c[i:i + CHUNK], ctx[i:i + CHUNK]

        c_emb = center_emb[c_chunk]          # (chunk, 64)
        p_emb = context_emb[ctx_chunk]       # (chunk, 64)

        pos_score = (c_emb * p_emb).sum(dim=1)
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_score, torch.ones_like(pos_score))

        neg_idx = torch.randint(0, n_items, (len(c_chunk), neg), device=device)
        neg_emb = context_emb[neg_idx]       # (chunk, neg, 64)
        neg_scores = (c_emb.unsqueeze(1) * neg_emb).sum(dim=2)
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores))

        loss = pos_loss + neg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_chunks += 1
    return total_loss / max(n_chunks, 1)


# ═══════════════════════════════════════════════════════════════════════
# STEP 4 — Evaluation (CSR-based; no giant Python structures)
# ═══════════════════════════════════════════════════════════════════════

def evaluate(embeddings, offsets, data, user_ids, user_index, train_counts,
             user_test, n_items, seed=SEED):
    """Candidate-generation protocol on the test split.

    Cohort: up to EVAL_SAMPLE users with >= 10 train events and >= 1 test item.
    Context: mean of last EVAL_CTX_LEN train embeddings (uniform).
    Score: cosine sim vs all items; repeats NOT excluded.
    Baselines: popularity (global count desc), user-frequency (user freq
    desc, ties by global popularity).
    """
    log("Evaluation")
    norm_emb = embeddings / (
        np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    pop_rank = np.argsort(-train_counts)              # global popularity order
    pop_pos = np.empty(n_items, dtype=np.int64)
    pop_pos[pop_rank] = np.arange(n_items)

    # Cohort
    eligible = [int(u) for i, u in enumerate(user_ids.tolist())
                if (offsets[i + 1] - offsets[i]) >= 10 and user_test.get(u)]
    rng = np.random.default_rng(seed)
    if len(eligible) > EVAL_SAMPLE:
        idx = rng.choice(len(eligible), EVAL_SAMPLE, replace=False)
        eval_users = [eligible[j] for j in idx]
    else:
        eval_users = eligible
    log(f"  cohort: {len(eval_users)} users (of {len(eligible)} eligible)")

    results = {"item2vec": {"overall": [], "repeat": [], "discovery": []},
               "popularity": {"overall": [], "repeat": [], "discovery": []},
               "user_frequency": {"overall": [], "repeat": [], "discovery": []}}

    for u in eval_users:
        i = user_index[u]
        seq = data[offsets[i]:offsets[i + 1]]         # user's train events
        test_items = user_test[u]
        if not len(seq) or not test_items:
            continue

        train_unique = np.unique(seq)
        train_set = set(train_unique.tolist())
        repeat_items = test_items & train_set
        discovery_items = test_items - train_set

        # ── item2vec: mean of last-50 embeddings ──
        ctx_items = seq[-EVAL_CTX_LEN:]
        ctx_vec = norm_emb[ctx_items].mean(axis=0)
        nrm = np.linalg.norm(ctx_vec)
        if nrm > 1e-8:
            ctx_vec = ctx_vec / nrm
            scores = norm_emb @ ctx_vec
            ranked_model = np.argsort(-scores)
            _add_metrics(results["item2vec"], ranked_model, test_items,
                         repeat_items, discovery_items)

        # ── popularity baseline (same ranking for everyone) ──
        _add_metrics(results["popularity"], pop_rank, test_items,
                     repeat_items, discovery_items)

        # ── user-frequency baseline: user freq desc, ties by popularity asc ──
        uf = np.bincount(seq, minlength=n_items)
        user_items = np.where(uf > 0)[0]
        order1 = user_items[np.lexsort((pop_pos[user_items], -uf[user_items]))]
        rest_mask = np.ones(n_items, dtype=bool)
        rest_mask[user_items] = False
        ranked_recent = np.concatenate(
            [order1, pop_rank[rest_mask[pop_rank]]])
        _add_metrics(results["user_frequency"], ranked_recent, test_items,
                     repeat_items, discovery_items)

    return {m: _aggregate(v) for m, v in results.items()}


def _add_metrics(bucket, ranked, test_items, repeat_items, discovery_items):
    o = _compute_metrics(ranked, test_items)
    bucket["overall"].append(o)
    if repeat_items:
        bucket["repeat"].append(_compute_metrics(ranked, repeat_items))
    if discovery_items:
        bucket["discovery"].append(_compute_metrics(ranked, discovery_items))


def _compute_metrics(ranked, test_items):
    """Recall@K, MRR@K, HitRate@K for one user against one ranking."""
    test_set = set(test_items)
    result = {}
    for k in RECALL_KS:
        top_k = set(ranked[:k].tolist())
        result[f"recall@{k}"] = (len(top_k & test_set) / len(test_set)
                                 if test_set else 0.0)
    rr = 0.0
    for rank, item in enumerate(ranked[:MRR_K]):
        if int(item) in test_set:
            rr = 1.0 / (rank + 1)
            break
    result[f"mrr@{MRR_K}"] = rr
    top_k = set(ranked[:HITRATE_K].tolist())
    result[f"hitrate@{HITRATE_K}"] = 1.0 if (top_k & test_set) else 0.0
    return result


def _aggregate(per_user_metrics):
    agg = {}
    for split in ["overall", "repeat", "discovery"]:
        rows = per_user_metrics[split]
        if not rows:
            agg[split] = {}
            continue
        agg[split] = {k: round(float(np.mean([m[k] for m in rows])), 6)
                      for k in rows[0]}
        agg[split]["n_users"] = len(rows)
    return agg


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def run_item2vec(events_path, vocab_path, output_dir):
    """Full pipeline: load → pairs → train → eval → reports."""
    output_dir = Path(output_dir)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)

    n_items = load_n_items(vocab_path)
    log(f"Vocab size (from vocab.parquet): {n_items:,}")

    offsets, data, user_ids, user_index, train_counts, user_test = \
        load_csr(events_path, n_items)

    log(f"  expected users={EXPECTED['n_users']:,}  actual={len(user_ids):,}")
    log(f"  expected items={EXPECTED['n_items']:,}  actual={n_items:,}")
    log(f"  train events={len(data):,} "
        f"(upper bound est. {EXPECTED['train_rows_upper']:,})")

    keep_probs = compute_keep_probs(train_counts, SUBSAMPLE_T)
    log(f"  subsampling: {int((keep_probs < 0.5).sum()):,} items with "
        f"keep < 0.5 (t={SUBSAMPLE_T} — near no-op on this corpus, see notes)")

    log("STEP 3: SGNS Training")
    embeddings = train_sgns(offsets, data, keep_probs, n_items, output_dir)

    log("STEP 4: Evaluation")
    results = evaluate(embeddings, offsets, data, user_ids, user_index,
                       train_counts, user_test, n_items)

    log("STEP 5: Writing reports")
    _write_reports(results, output_dir)

    log("DONE")
    return results


def _write_reports(results, output_dir):
    """Write metrics.json, eval_report.md, README.md."""
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"  wrote {metrics_path}")

    metric_keys = [f"recall@{RECALL_KS[0]}", f"recall@{RECALL_KS[1]}",
                   f"mrr@{MRR_K}", f"hitrate@{HITRATE_K}"]
    header = "| Model | Split | " + " | ".join(metric_keys) + " |"
    sep = "|-------|-------|" + "|".join(["------"] * len(metric_keys)) + "|"
    lines = ["# Item2Vec Evaluation Report", "",
             f"**Generated:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
             "**Kernel:** lb_item2vec.py", "", "## Metrics", "", header, sep]
    for model, splits in results.items():
        for split in ["overall", "repeat", "discovery"]:
            r = splits.get(split)
            if r:
                vals = [f"{r.get(mk, 0.0):.4f}" for mk in metric_keys]
                lines.append(f"| {model} | {split} (n={r.get('n_users', 0)}) | "
                             + " | ".join(vals) + " |")
    lines += ["", "## Notes", "",
              "- **Context:** mean of last 50 train embeddings (uniform).",
              "- **Negatives:** uniform random from vocab (v1).",
              "- **Eval:** repeats NOT excluded (legitimate predictions).",
              "- **Subsampling:** t=1e-3 — effectively no-op on this corpus "
              "(only items with >~2M listens affected).",
              "- **Baselines:** popularity = global count desc; "
              "user_frequency = user freq desc, ties by popularity.",
              "- **Eval type:** set-based — recall/MRR against the full "
              "test-set of held-out items, not next-item prediction.",
              "- **Primary metric:** discovery (items the user has not "
              "listened to); overall recall is dominated by repeat "
              "consumption."]
    (output_dir / "reports" / "eval_report.md").write_text(
        "\n".join(lines), encoding="utf-8")
    log("  wrote reports/eval_report.md")

    readme = """# Music Recommender — Item2Vec Embeddings

SGNS (Skip-Gram with Negative Sampling) embeddings for MLHD+ listening history.

## Files

| File | Description |
|------|-------------|
| `models/item2vec_final.npy` | Final center embeddings (f32, [vocab_size, 64]) |
| `models/item2vec_e{N}.npy` | Per-epoch checkpoints |
| `metrics.json` | Evaluation metrics vs baselines |

Join with `vocab.parquet` from lb-trainprep output on `item_id` (row index).

**Rerun:** Kaggle GPU kernel (T4). CPU fallback works but is slow.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    log("  wrote README.md")

    summary = {m: {s: {k: results[m][s].get(k)
                       for k in metric_keys if k in results[m][s]}
                   for s in ["overall", "repeat", "discovery"]
                   if results[m].get(s)}
               for m in results}
    print(json.dumps(summary))


def main():
    """Kaggle entry point — discover inputs, run full pipeline."""
    root = Path("/kaggle/input")
    events_files = list(root.rglob("events.parquet"))
    vocab_files = list(root.rglob("vocab.parquet"))

    tree = "\n".join(f"  {p}" for p in sorted(root.rglob("*")) if p.is_file())
    log(f"/kaggle/input tree:\n{tree}")

    if not events_files:
        sys.exit("FATAL: events.parquet not found under /kaggle/input")
    if len(events_files) > 1:
        sys.exit(f"FATAL: multiple events.parquet found, expected exactly one: {events_files}")
    if not vocab_files:
        sys.exit("FATAL: vocab.parquet not found under /kaggle/input")
    if len(vocab_files) > 1:
        sys.exit(f"FATAL: multiple vocab.parquet found, expected exactly one: {vocab_files}")

    log(f"Events: {events_files[0]}")
    log(f"Vocab:  {vocab_files[0]}")
    run_item2vec(events_files[0], vocab_files[0], Path("/kaggle/working"))


if __name__ == "__main__":
    main()
