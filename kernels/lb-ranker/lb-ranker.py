"""
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Shuvi

GRU ranker kernel: sequential scoring over frozen item2vec embeddings.

Trains a GRU that reads a user's recent listening history and scores
candidate items. Frozen item2vec embeddings provide the input features
and the scoring space.

Input:  events.parquet + vocab.parquet from lb-trainprep,
        item2vec_final.npy from lb-item2vec
Output: models/ranker_final.pt, reports/ranker_report.md, metrics.json

Pipeline:
  Step 1 — Load data into CSR (per-user sequences, two vectorized passes)
  Step 2 — Sliding window pair generation (context → target, vectorized per user)
  Step 3 — GRU training (PyTorch, GPU if available)
  Step 4 — Evaluation (Recall@20/100, MRR@10, HitRate@20) + 3 baselines
  Step 5 — README.md

Design notes:
  - CSR layout: one int32 array + offsets. NO per-user Python lists
    (1.27B Python ints would be ~45GB RAM — Kaggle has 13GB).
  - Sliding window: context = up to SEQ_LEN items BEFORE position,
    target = item AT position. Left-pad with zero vector for short contexts.
  - GRU hidden_size=128, Linear(128→64) projection, dot-product score.
  - Sampled softmax: candidates = target + in-batch targets + NEG_POP
    items (prob ∝ sqrt(train_count)). ~1024 candidates per window.
  - Per-user window cap (100) limits heavy-user dominance in training —
    no extra weighting (noted in report).
  - Frozen embeddings: E input rows have .detach(), no grad through E.
"""
import json, os, sys, time
from collections import defaultdict
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────
EMB_DIM         = 64
GRU_HIDDEN      = 128
GRU_LAYERS      = 1
SEQ_LEN         = 50            # max context length (last N train items)
BATCH_WINDOWS   = 256           # windows per training batch (after padding)
CHUNK           = 10_000        # sub-batch for sampled softmax memory
NEG_POP         = 1024          # popularity-biased negatives per window
EPOCHS          = 2
LR              = 1e-3
SEED            = 42
LOG_INTERVAL    = 100           # log every N optimizer steps
EVAL_SAMPLE     = 5_000         # users to sample for eval
EVAL_CTX_LEN    = 50            # last N train events for eval context
RECALL_KS       = [20, 100]
MRR_K           = 10
HITRATE_K       = 20
MAX_USER_WINDOWS = 100          # per-user window cap (limit heavy-user dominance)

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


def load_embeddings(npy_path, n_items):
    """Load frozen item2vec embeddings. Returns (n_items, 64) float32 numpy."""
    emb = np.load(str(npy_path)).astype(np.float32)
    assert emb.shape[1] == EMB_DIM, f"Expected dim {EMB_DIM}, got {emb.shape}"
    if emb.shape[0] < n_items:
        pad = np.zeros((n_items - emb.shape[0], EMB_DIM), dtype=np.float32)
        emb = np.vstack([emb, pad])
    elif emb.shape[0] > n_items:
        emb = emb[:n_items]
    return emb


# ═══════════════════════════════════════════════════════════════════════
# STEP 2 — Sliding window pair generation (vectorized per user)
# ═══════════════════════════════════════════════════════════════════════

def build_windows_for_user(seq, seq_len, max_windows, rng):
    """Generate sliding-window (context, target) pairs for one user.

    For each position i in seq:
      context = seq[max(0, i-seq_len):i]   (up to seq_len items before i)
      target  = seq[i]

    Right-padded: valid items at START, zeros at END. This lets
    pack_padded_sequence work correctly in the GRU forward pass.

    Caps at max_windows per user. Preallocates arrays.
    Returns: (contexts, targets, lengths) where
      contexts: int32 (n_windows, seq_len)  — right-padded
      targets:  int32 (n_windows,)
      lengths:  int32 (n_windows,)          — actual context length
    """
    n = len(seq)
    if n < 2:
        return (np.zeros((0, seq_len), dtype=np.int32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32))

    n_windows = min(n - 1, max_windows)  # positions start at 1: pos 0 has no context
    # Sample window positions if user is long
    if n_windows < n - 1:
        positions = np.sort(rng.choice(np.arange(1, n), n_windows, replace=False))
    else:
        positions = np.arange(1, n)

    contexts = np.zeros((n_windows, seq_len), dtype=np.int32)
    targets = np.empty(n_windows, dtype=np.int32)
    lengths = np.empty(n_windows, dtype=np.int32)

    for w, pos in enumerate(positions):
        ctx_start = max(0, int(pos) - seq_len)
        ctx_len = int(pos) - ctx_start
        if ctx_len > 0:
            contexts[w, :ctx_len] = seq[ctx_start:int(pos)]
        targets[w] = seq[int(pos)]
        lengths[w] = ctx_len

    return contexts, targets, lengths


def build_all_windows(offsets, data, user_ids, seq_len=SEQ_LEN,
                      max_windows=MAX_USER_WINDOWS, seed=SEED):
    """Build training windows for all users. Returns concatenated arrays.

    Returns:
      all_ctx:   int32 (N, seq_len)
      all_tgt:   int32 (N,)
      all_len:   int32 (N,)
    """
    rng = np.random.default_rng(seed)
    parts_ctx, parts_tgt, parts_len = [], [], []

    for i, uid in enumerate(user_ids.tolist()):
        seq = data[offsets[i]:offsets[i + 1]]
        if len(seq) < 2:
            continue
        c, t, l = build_windows_for_user(seq, seq_len, max_windows, rng)
        if len(c) > 0:
            parts_ctx.append(c)
            parts_tgt.append(t)
            parts_len.append(l)

        if (i + 1) % 5000 == 0:
            total = sum(len(p) for p in parts_ctx)
            log(f"  windows: {i + 1:>5}/{len(user_ids)} users, "
                f"{total:>12,} windows")

    all_ctx = np.concatenate(parts_ctx) if parts_ctx else np.zeros((0, seq_len), dtype=np.int32)
    all_tgt = np.concatenate(parts_tgt) if parts_tgt else np.zeros(0, dtype=np.int32)
    all_len = np.concatenate(parts_len) if parts_len else np.zeros(0, dtype=np.int32)

    log(f"  total windows: {len(all_ctx):,}")
    return all_ctx, all_tgt, all_len


# ═══════════════════════════════════════════════════════════════════════
# STEP 3 — GRU Training
# ═══════════════════════════════════════════════════════════════════════

class GRURanker:
    """GRU over frozen embeddings + Linear projection + dot-product scoring."""

    def __init__(self, n_items, emb_dim=EMB_DIM, hidden=GRU_HIDDEN,
                 layers=GRU_LAYERS, device="cpu"):
        import torch
        import torch.nn as nn
        self.device = torch.device(device)
        self.emb = nn.Embedding(n_items, emb_dim).to(self.device)
        self.gru = nn.GRU(emb_dim, hidden, num_layers=layers,
                          batch_first=True).to(self.device)
        self.proj = nn.Linear(hidden, emb_dim).to(self.device)
        self.emb_dim = emb_dim
        self.n_items = n_items

    def forward(self, ctx_ids, lengths):
        """Process right-padded context sequences.

        Right-padded: valid items are at the START of each row,
        padding at the END. pack_padded_sequence processes the first
        `length` elements → h_n is the correct hidden state.

        Args:
          ctx_ids: (B, seq_len) int32 item_ids (right-padded)
          lengths: (B,) int32 actual context lengths
        Returns:
          h: (B, emb_dim) — projected hidden state for each window
        """
        import torch

        x = self.emb(ctx_ids.long())       # (B, seq_len, 64)
        seq_len = x.size(1)
        lengths_cpu = lengths.cpu().long().clamp(min=1, max=seq_len)

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)           # h_n: (layers, B, hidden)
        h = h_n[-1]                         # (B, hidden)
        h = self.proj(h)                    # (B, 64)
        return h

    def score(self, h, item_ids):
        """Dot product of h vs item embeddings.

        Args:
          h: (B, 64)
          item_ids: (B,) or (B, K) int32
        Returns:
          logits: (B,) or (B, K)
        """
        import torch
        e = self.emb(item_ids.long())
        return (h * e).sum(dim=-1)

    def state_dict(self):
        return {"emb": self.emb, "gru": self.gru, "proj": self.proj}

    def save(self, path):
        import torch
        torch.save({k: v.state_dict() for k, v in self.state_dict().items()}, str(path))

    def load(self, path):
        import torch
        ckpt = torch.load(str(path), map_location=self.device, weights_only=True)
        for k, sd in ckpt.items():
            self.state_dict()[k].load_state_dict(sd)


def build_neg_pool(train_counts, n_items, seed=SEED):
    """Precompute popularity-biased sampling distribution.

    Prob ∝ sqrt(count). Items with count=0 get zero prob.
    """
    sqrt_counts = np.sqrt(np.maximum(train_counts, 0).astype(np.float64))
    total = sqrt_counts.sum()
    if total == 0:
        probs = np.ones(n_items, dtype=np.float64) / n_items
    else:
        probs = sqrt_counts / total
    return probs


def train_ranker(all_ctx, all_tgt, all_len, n_items, neg_probs,
                 embeddings, output_dir, epochs=EPOCHS, lr=LR, seed=SEED):
    """Train GRU ranker with sampled softmax loss. Returns trained GRURanker."""
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Training GRU ranker: device={device}, {n_items:,} items")

    model = GRURanker(n_items, EMB_DIM, GRU_HIDDEN, GRU_LAYERS, str(device))

    # Copy frozen item2vec embeddings into model.emb and freeze
    model.emb.weight.data = torch.from_numpy(embeddings).to(model.device)
    model.emb.weight.requires_grad_(False)

    optimizer = torch.optim.Adam(
        list(model.gru.parameters()) + list(model.proj.parameters()), lr=lr)

    output_dir = Path(output_dir)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)

    n_total = len(all_ctx)
    rng_neg = np.random.default_rng(seed)

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        epoch_loss = 0.0
        n_steps = 0
        epoch_samples = 0

        # Shuffle window order
        order = np.arange(n_total)
        np.random.default_rng(seed + epoch).shuffle(order)

        for start in range(0, n_total, BATCH_WINDOWS):
            idx = order[start:start + BATCH_WINDOWS]
            if len(idx) < 2:
                continue

            ctx_batch = torch.from_numpy(all_ctx[idx]).long().to(device)
            len_batch = torch.from_numpy(all_len[idx]).long().to(device)
            tgt_batch = torch.from_numpy(all_tgt[idx]).long().to(device)

            # Forward: get hidden states
            h = model.forward(ctx_batch, len_batch)  # (B, 64)

            # Build candidate set: target + in-batch negatives + pop negatives
            B = len(idx)
            pos_ids = tgt_batch.unsqueeze(1)           # (B, 1)

            # In-batch negatives: each target as negative for others
            inbatch_neg = tgt_batch.unsqueeze(0).expand(B, B)  # (B, B)
            # Mask diagonal (target is not its own negative)
            mask = ~torch.eye(B, dtype=torch.bool, device=device)
            inbatch_neg = inbatch_neg[mask].view(B, B - 1)     # (B, B-1)

            # Popularity negatives
            pop_ids_np = rng_neg.choice(n_items, size=(B, NEG_POP), p=neg_probs)
            pop_neg = torch.from_numpy(pop_ids_np).long().to(device)

            # All candidates: [target, inbatch_neg, pop_neg]
            candidates = torch.cat([pos_ids, inbatch_neg, pop_neg], dim=1)  # (B, K)
            K = candidates.size(1)

            # Compute logits
            cand_emb = model.emb(candidates)                # (B, K, 64)
            logits = (h.unsqueeze(1) * cand_emb).sum(dim=2) # (B, K)

            # Cross-entropy: target is index 0 in candidates
            labels = torch.zeros(B, dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_steps += 1
            epoch_samples += B

            if n_steps % LOG_INTERVAL == 0:
                el = time.time() - t_epoch
                avg_loss = epoch_loss / n_steps
                log(f"  step {n_steps:>5}  samples={epoch_samples:>13,}  "
                    f"loss={avg_loss:.4f}  samples/s={epoch_samples / max(el, 0.001):,.0f}")

        elapsed = time.time() - t_epoch
        avg_loss = epoch_loss / max(n_steps, 1)
        log(f"  epoch {epoch} done: {epoch_samples:,} samples, "
            f"avg_loss={avg_loss:.4f}, {elapsed:.0f}s")

        model.save(output_dir / "models" / f"ranker_e{epoch}.pt")

    model.save(output_dir / "models" / "ranker_final.pt")
    log(f"  saved models/ranker_final.pt")
    return model


# ═══════════════════════════════════════════════════════════════════════
# STEP 4 — Evaluation (CSR-based; no giant Python structures)
# ═══════════════════════════════════════════════════════════════════════

def evaluate(model, embeddings, offsets, data, user_ids, user_index,
             train_counts, user_test, n_items, seed=SEED):
    """Candidate-generation protocol on the test split.

    Cohort: up to EVAL_SAMPLE users with >= 10 train events and >= 1 test item.
    Context: last EVAL_CTX_LEN train items → GRU → score all items.
    Also scores: item2vec (mean embedding), popularity, user-frequency baselines.
    Repeats NOT excluded.
    """
    import torch

    log("Evaluation (4 scorers)")
    device = model.device
    norm_emb = embeddings / (
        np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    pop_rank = np.argsort(-train_counts)
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

    results = {m: {"overall": [], "repeat": [], "discovery": []}
               for m in ["ranker", "item2vec", "popularity", "user_frequency"]}

    # Pre-embed all items on device for ranker scoring
    with torch.no_grad():
        all_emb_t = torch.from_numpy(embeddings).to(device)  # (n_items, 64)

    for u in eval_users:
        i = user_index[u]
        seq = data[offsets[i]:offsets[i + 1]]
        test_items = user_test[u]
        if not len(seq) or not test_items:
            continue

        train_unique = np.unique(seq)
        train_set = set(train_unique.tolist())
        repeat_items = test_items & train_set
        discovery_items = test_items - train_set

        # ── ranker: GRU over last EVAL_CTX_LEN items (right-padded) ──
        ctx_items = seq[-EVAL_CTX_LEN:]
        ctx_len = len(ctx_items)
        ctx_padded = np.zeros(EVAL_CTX_LEN, dtype=np.int32)
        ctx_padded[:ctx_len] = ctx_items
        with torch.no_grad():
            ctx_t = torch.from_numpy(ctx_padded).long().unsqueeze(0).to(device)
            lens = torch.tensor([ctx_len], dtype=torch.long, device=device)
            h = model.forward(ctx_t, lens)          # (1, 64)
            h_norm = h / (h.norm(dim=1, keepdim=True) + 1e-8)
            scores_r = (all_emb_t @ h_norm.squeeze(0)).cpu().numpy()
        ranked_ranker = np.argsort(-scores_r)
        _add_metrics(results["ranker"], ranked_ranker, test_items,
                     repeat_items, discovery_items)

        # ── item2vec: mean of last-50 embeddings ──
        ctx_vec = norm_emb[ctx_items].mean(axis=0)
        nrm = np.linalg.norm(ctx_vec)
        if nrm > 1e-8:
            ctx_vec = ctx_vec / nrm
            scores_i2v = norm_emb @ ctx_vec
            ranked_i2v = np.argsort(-scores_i2v)
            _add_metrics(results["item2vec"], ranked_i2v, test_items,
                         repeat_items, discovery_items)

        # ── popularity baseline ──
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

def run_ranker(events_path, vocab_path, emb_path, output_dir):
    """Full pipeline: load → windows → train → eval → reports."""
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

    embeddings = load_embeddings(emb_path, n_items)
    log(f"  embeddings loaded: {embeddings.shape}")

    neg_probs = build_neg_pool(train_counts, n_items)
    log(f"  neg pool: {np.sum(neg_probs > 0):,} items with nonzero prob")

    log("STEP 2: Sliding window pair generation")
    all_ctx, all_tgt, all_len = build_all_windows(
        offsets, data, user_ids, SEQ_LEN, MAX_USER_WINDOWS, SEED)

    log("STEP 3: GRU Training")
    model = train_ranker(all_ctx, all_tgt, all_len, n_items, neg_probs,
                         embeddings, output_dir)

    log("STEP 4: Evaluation")
    results = evaluate(model, embeddings, offsets, data, user_ids,
                       user_index, train_counts, user_test, n_items)

    log("STEP 5: Writing reports")
    _write_reports(results, output_dir)

    log("DONE")
    return results


def _write_reports(results, output_dir):
    """Write metrics.json, ranker_report.md, README.md."""
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"  wrote {metrics_path}")

    metric_keys = [f"recall@{RECALL_KS[0]}", f"recall@{RECALL_KS[1]}",
                   f"mrr@{MRR_K}", f"hitrate@{HITRATE_K}"]
    header = "| Model | Split | " + " | ".join(metric_keys) + " |"
    sep = "|-------|-------|" + "|".join(["------"] * len(metric_keys)) + "|"
    lines = ["# GRU Ranker Evaluation Report", "",
             f"**Generated:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
             "**Kernel:** lb_ranker.py", "",
             "## Metrics", "", header, sep]
    for model, splits in results.items():
        for split in ["overall", "repeat", "discovery"]:
            r = splits.get(split)
            if r:
                vals = [f"{r.get(mk, 0.0):.4f}" for mk in metric_keys]
                lines.append(f"| {model} | {split} (n={r.get('n_users', 0)}) | "
                             + " | ".join(vals) + " |")
    lines += ["", "## Architecture", "",
              "- **Model:** GRU(input=64, hidden=128) + Linear(128→64) + dot",
              "- **Input:** frozen item2vec embeddings (detached, no grad)",
              "- **Loss:** sampled softmax (~1024 candidates: target + "
              "in-batch + popularity-biased negatives)",
              "- **Per-user window cap:** 100 (limits heavy-user dominance "
              "in training)",
              "", "## Notes", "",
              "- **Context:** last 50 train items (right-padded to SEQ_LEN)",
              "- **Eval:** repeats NOT excluded (legitimate predictions)",
              "- **Eval type:** set-based — recall/MRR against the full "
              "test-set of held-out items, not next-item prediction",
              "- **Primary metric:** discovery (items the user has not "
              "listened to); overall recall is dominated by repeat "
              "consumption",
              "- **Negatives:** in-batch negatives can collide with true "
              "targets across users (sampled-softmax approximation)",
              "- **Baselines:** item2vec = mean embedding; popularity = "
              "global count desc; user_frequency = user freq desc, ties "
              "by popularity",
              "- **Negatives:** prob ∝ sqrt(train_count)"]
    (output_dir / "reports" / "ranker_report.md").write_text(
        "\n".join(lines), encoding="utf-8")
    log("  wrote reports/ranker_report.md")

    readme = """# Music Recommender — GRU Ranker

Sequential ranker over frozen item2vec embeddings for MLHD+ listening history.

## Files

| File | Description |
|------|-------------|
| `models/ranker_final.pt` | Final model state_dict |
| `models/ranker_e{N}.pt` | Per-epoch checkpoints |
| `metrics.json` | Evaluation metrics vs baselines |

## Consuming Model

```python
import torch, numpy as np, pyarrow.parquet as pq

# Load embeddings
emb = np.load("item2vec_final.npy")  # (vocab_size, 64) float32

# Load model
ckpt = torch.load("models/ranker_final.pt", weights_only=True)
# ckpt["emb"], ckpt["gru"], ckpt["proj"] are state_dicts
```

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
    emb_files = list(root.rglob("item2vec_final.npy"))

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
    if not emb_files:
        sys.exit("FATAL: item2vec_final.npy not found under /kaggle/input")

    log(f"Events: {events_files[0]}")
    log(f"Vocab:  {vocab_files[0]}")
    log(f"Emb:    {emb_files[0]}")
    run_ranker(events_files[0], vocab_files[0], emb_files[0],
               Path("/kaggle/working"))


if __name__ == "__main__":
    main()
