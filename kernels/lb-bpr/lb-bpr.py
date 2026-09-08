"""
lb_bpr.py — BPR-MF implicit-feedback baseline kernel (Paper 2).

Trains a Bayesian Personalized Ranking matrix factorization (Rendle et al.,
2009) on the sanitized complete-tier listening events, as the classical
implicit-feedback baseline for Paper 2's recommender comparison.

Input (from lb-trainprep):
  - events.parquet  (columns: user, item_id, split; user-grouped, ts-ordered)
  - vocab.parquet   (column: item_id; dense 0..N-1)

Output:
  - models/bpr_final.pt        (user + item factor state_dicts)
  - reports/bpr_report.md      (metrics table + architecture + notes)
  - metrics.json               (machine-readable metrics)
  - README.md

Pipeline:
  1. Load CSR train matrix (split==0) + per-user test sets (split==2).
  2. Train BPR-MF with sampled (user, positive, negative) triples.
  3. Evaluate on the same protocol as lb_ranker / lb_item2vec:
     recall@20, recall@100, mrr@10, hitrate@20 — overall / repeat / discovery.
  4. Write reports.

Design notes:
  - Same eval cohort + seed as lb_ranker (EVAL_SAMPLE=5000, SEED=42) so
    numbers are directly comparable across kernels.
  - Popularity and user-frequency baselines are re-evaluated here (no
    external artifacts needed). Ranker/item2vec numbers come from their
    own kernels under the identical protocol.
  - Negatives: uniform over items, resampled up to 10 times if they fall
    in the user's train set (classic BPR exclusion); accepted otherwise.
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

T0 = time.time()


def log(m):
    print(f"[{int(time.time() - T0):>5}s] {m}", flush=True)


CONFIG = {
    "D": 64,                # factor dimension (matches EMB_DIM convention)
    "EPOCHS": 3,
    "LR": 1e-3,              # Adam, consistent with lb_ranker
    "BATCH_TRIPLES": 4096,
    "PER_USER_CAP": 100,     # max positive triples sampled per user per epoch
    "NEG_RETRIES": 10,       # resample attempts for negative exclusion
    "SEED": 42,
    "LOG_INTERVAL": 100,
    "EVAL_SAMPLE": 5_000,
    "RECALL_KS": [20, 100],
    "MRR_K": 10,
    "HITRATE_K": 20,
    "MIN_TRAIN_EVENTS": 10,  # eval cohort eligibility
}

EXPECTED = {
    "n_users": 36_970,
    "n_items": 2_803_656,
    "train_rows_upper": 1_270_000_000,
}


def _ensure_torch_gpu():
    """Install a P100-compatible torch build before importing torch."""
    import subprocess

    try:
        cap = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except Exception:
        return  # no GPU tooling — CPU torch is fine
    if cap.startswith("6."):
        log("P100 (sm_60) detected — installing cu118 torch build")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "torch==2.4.1+cu118",
             "--index-url", "https://download.pytorch.org/whl/cu118"],
            check=False,
        )


_ensure_torch_gpu()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading (verbatim protocol from lb_ranker)
# ---------------------------------------------------------------------------

def load_csr(events_path, n_items):
    """Two vectorized passes over events.parquet -> CSR train matrix + test sets.

    split==0 rows are train; split==2 rows are test (per-user item sets).
    The file must be user-grouped (trainprep guarantees this).
    """
    user_train_counts = defaultdict(int)
    user_test = defaultdict(set)
    user_regressions = 0

    log(f"pass 1: scanning {events_path}")
    pf = pq.ParquetFile(events_path)
    batch_i = 0
    for batch in pf.iter_batches(batch_size=2_000_000, columns=["user", "item_id", "split"]):
        u = batch.column("user").to_numpy()
        i = batch.column("item_id").to_numpy()
        s = batch.column("split").to_numpy()
        if len(u) and (user_regressions := int(np.sum(u[1:] < u[:-1]))):
            user_regressions += user_regressions
        train_mask = s == 0
        if train_mask.any():
            ut, it = u[train_mask], i[train_mask]
            for uu, ii in zip(ut.tolist(), it.tolist()):
                user_train_counts[uu] += 1
        test_mask = s == 2
        if test_mask.any():
            ut, it = u[test_mask], i[test_mask]
            for uu, ii in zip(ut.tolist(), it.tolist()):
                user_test[uu].add(ii)
        batch_i += 1
        if batch_i % 50 == 0:
            log(f"  batch {batch_i}: users={len(user_train_counts)}")
    if user_regressions:
        log(f"FATAL: {user_regressions} user-order regressions — file not user-grouped")
        sys.exit(1)

    user_ids = np.array(sorted(user_train_counts), dtype=np.int32)
    n_users = len(user_ids)
    user_index = {int(u): i for i, u in enumerate(user_ids)}
    offsets = np.zeros(n_users + 1, dtype=np.int64)
    for i, u in enumerate(user_ids):
        offsets[i + 1] = offsets[i] + user_train_counts[int(u)]
    total = int(offsets[-1])
    log(f"pass 1 done: n_users={n_users} train_rows={total} test_users={len(user_test)}")

    log("pass 2: filling CSR data")
    data = np.empty(total, dtype=np.int32)
    fill = defaultdict(int)
    batch_i = 0
    for batch in pf.iter_batches(batch_size=2_000_000, columns=["user", "item_id", "split"]):
        u = batch.column("user").to_numpy()
        i = batch.column("item_id").to_numpy()
        s = batch.column("split").to_numpy()
        train_mask = s == 0
        ut, it = u[train_mask], i[train_mask]
        for uu, ii in zip(ut.tolist(), it.tolist()):
            k = user_index[uu]
            data[offsets[k] + fill[k]] = ii
            fill[k] += 1
        batch_i += 1
        if batch_i % 50 == 0:
            log(f"  batch {batch_i}")
    if any(fill[k] != (offsets[k + 1] - offsets[k]) for k in range(n_users)):
        log("FATAL: CSR fill mismatch")
        sys.exit(1)

    train_counts = np.bincount(data, minlength=n_items).astype(np.int64)
    log(f"CSR ready: data={data.shape[0]} items_seen={(train_counts > 0).sum()}")
    return offsets, data, user_ids, user_index, train_counts, user_test


def load_n_items(vocab_path):
    t = pq.read_table(vocab_path, columns=["item_id"])
    n = int(pq.read_table(vocab_path, columns=["item_id"]).column("item_id").to_numpy().max()) + 1
    log(f"vocab: n_items={n}")
    return n


# ---------------------------------------------------------------------------
# BPR-MF model + training
# ---------------------------------------------------------------------------

class BPRMF(nn.Module):
    def __init__(self, n_users, n_items, d, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.user_factors = nn.Embedding(n_users, d)
        self.item_factors = nn.Embedding(n_items, d)
        with torch.no_grad():
            self.user_factors.weight.copy_(torch.randn(n_users, d, generator=g) * 0.01)
            self.item_factors.weight.copy_(torch.randn(n_items, d, generator=g) * 0.01)

    def score_pairs(self, u, i):
        uf = self.user_factors(u)
        inf = self.item_factors(i)
        return (uf * inf).sum(dim=1)


def sample_triples(offsets, data, n_items, n_users, cap, neg_retries, batch_size, rng):
    """Yield batches of (u_idx, pos, neg) triples.

    Per user per epoch: min(train_events, cap) positives. Negatives are
    uniform over all items, resampled up to neg_retries times if they fall
    in the user's train set.
    """
    per_user = np.minimum(offsets[1:] - offsets[:-1], cap)
    total = int(per_user.sum())
    user_of_triple = np.repeat(np.arange(n_users), per_user)
    order = rng.permutation(total)

    # Pre-extract per-user train sets lazily via CSR slices.
    ptr = 0
    while ptr < len(order):
        idx = order[ptr:ptr + batch_size]
        ptr += batch_size
        u_idx = user_of_triple[idx]
        # position within the user's (capped) positive list
        starts = offsets[u_idx]
        counts = offsets[u_idx + 1] - starts
        capped_counts = np.minimum(counts, cap)
        base = idx - np.concatenate(([0], np.cumsum(per_user)))[user_of_triple[idx]] \
            if False else None  # unused; computed below
        # simpler: offset of this triple within the user's capped list
        cum_before = np.concatenate(([0], np.cumsum(per_user)))
        within = idx - cum_before[user_of_triple[idx]]
        pos = data[starts + within]
        neg = rng.integers(0, n_items, size=len(idx))
        for _ in range(neg_retries):
            bad = np.zeros(len(idx), dtype=bool)
            for k in range(len(idx)):
                s = offsets[u_idx[k]]
                e = offsets[u_idx[k] + 1]
                if neg[k] in data[s:e]:
                    bad[k] = True
            if not bad.any():
                break
            neg[bad] = rng.integers(0, n_items, size=int(bad.sum()))
        yield u_idx, pos, neg


def train_bpr(offsets, data, n_users, n_items, output_dir, cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"training on {device}")
    model = BPRMF(n_users, n_items, cfg["D"], cfg["SEED"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["LR"])
    rng = np.random.default_rng(cfg["SEED"])

    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["EPOCHS"]):
        model.train()
        total_loss, n_batches = 0.0, 0
        gen = sample_triples(
            offsets, data, n_items, n_users,
            cap=cfg["PER_USER_CAP"], neg_retries=cfg["NEG_RETRIES"],
            batch_size=cfg["BATCH_TRIPLES"], rng=rng,
        )
        for u_idx, pos, neg in gen:
            u_t = torch.from_numpy(u_idx.astype(np.int64)).to(device)
            p_t = torch.from_numpy(pos.astype(np.int64)).to(device)
            n_t = torch.from_numpy(neg.astype(np.int64)).to(device)
            opt.zero_grad()
            pos_score = model.score_pairs(u_t, p_t)
            neg_score = model.score_pairs(u_t, n_t)
            loss = -(torch.sigmoid(pos_score - neg_score).log()).mean()
            loss.backward()
            opt.step()
            total_loss += float(loss)
            n_batches += 1
            if n_batches % cfg["LOG_INTERVAL"] == 0:
                log(f"  epoch {epoch} batch {n_batches} loss={total_loss / n_batches:.4f}")
        log(f"epoch {epoch} done: loss={total_loss / max(n_batches, 1):.4f}")
        torch.save(model.state_dict(), models_dir / f"bpr_e{epoch}.pt")

    torch.save(model.state_dict(), models_dir / "bpr_final.pt")
    log("saved bpr_final.pt")
    return model, device


# ---------------------------------------------------------------------------
# Evaluation (same protocol as lb_ranker)
# ---------------------------------------------------------------------------

def _compute_metrics(ranked, test_items, cfg):
    ranked_set = set(ranked[:100].tolist())
    hits20 = len(ranked_set.intersection(test_items))
    hits100 = len(set(ranked[:100].tolist()).intersection(test_items))
    m = {
        f"recall@{cfg['RECALL_KS'][0]}": hits20 / len(test_items),
        f"recall@{cfg['RECALL_KS'][1]}": hits100 / len(test_items),
    }
    rr = 0.0
    for k in range(cfg["MRR_K"]):
        if ranked[k] in test_items:
            rr = 1.0 / (k + 1)
            break
    m[f"mrr@{cfg['MRR_K']}"] = rr
    m[f"hitrate@{cfg['HITRATE_K']}"] = 1.0 if hits20 > 0 else 0.0
    return m


def _add_metrics(bucket, ranked, test_items, repeat_items, discovery_items, cfg):
    bucket["overall"].append(_compute_metrics(ranked, test_items, cfg))
    if repeat_items:
        bucket["repeat"].append(_compute_metrics(ranked, repeat_items, cfg))
    if discovery_items:
        bucket["discovery"].append(_compute_metrics(ranked, discovery_items, cfg))


def _aggregate(results):
    out = {}
    for model, splits in results.items():
        out[model] = {}
        for split, rows in splits.items():
            if not rows:
                continue
            out[model][split] = {
                k: round(float(np.mean([r[k] for r in rows])), 6)
                for k in rows[0]
            }
            out[model][split]["n_users"] = len(rows)
    return out


def evaluate(model, device, offsets, data, user_index, train_counts, user_test,
             n_items, cfg):
    log("evaluating")
    pop_rank = np.argsort(-train_counts)
    pop_pos = np.empty(n_items, dtype=np.int64)
    pop_pos[pop_rank] = np.arange(n_items)

    eligible = [
        int(u) for u in user_index
        if (offsets[user_index[u] + 1] - offsets[user_index[u]]) >= cfg["MIN_TRAIN_EVENTS"]
        and user_test.get(int(u))
    ]
    rng = np.random.default_rng(cfg["SEED"])
    if len(eligible) > cfg["EVAL_SAMPLE"]:
        keep = rng.choice(len(eligible), cfg["EVAL_SAMPLE"], replace=False)
        eligible = [eligible[k] for k in keep]
    log(f"eval cohort: {len(eligible)} users")

    results = {
        "bpr_mf": {"overall": [], "repeat": [], "discovery": []},
        "popularity": {"overall": [], "repeat": [], "discovery": []},
        "user_frequency": {"overall": [], "repeat": [], "discovery": []},
    }

    model.eval()
    with torch.no_grad():
        item_matrix = model.item_factors.weight.detach().to(device)
        user_matrix = model.user_factors.weight.detach().to(device)

    for n, u in enumerate(eligible):
        i = user_index[u]
        seq = data[offsets[i]:offsets[i + 1]]
        test_items = user_test[int(u)]
        train_set = set(seq.tolist())
        repeat_items = test_items.intersection(train_set)
        discovery_items = test_items - train_set

        # BPR scorer: user factor @ all item factors
        with torch.no_grad():
            scores = item_matrix @ user_matrix[i]
            bpr_ranked = torch.argsort(scores, descending=True).cpu().numpy()
        _add_metrics(results["bpr_mf"], bpr_ranked, test_items,
                     repeat_items, discovery_items, cfg)

        # popularity baseline
        _add_metrics(results["popularity"], pop_rank, test_items,
                     repeat_items, discovery_items, cfg)

        # user_frequency baseline
        uf = np.bincount(seq, minlength=n_items)
        user_items = np.where(uf > 0)[0]
        order1 = user_items[np.lexsort((pop_pos[user_items], -uf[user_items]))]
        rest = pop_rank[~np.isin(pop_rank, user_items)]
        uf_ranked = np.concatenate([order1, rest])
        _add_metrics(results["user_frequency"], uf_ranked, test_items,
                     repeat_items, discovery_items, cfg)

        if (n + 1) % 500 == 0:
            log(f"  {n + 1}/{len(eligible)} users evaluated")

    return _aggregate(results)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _write_reports(metrics, output_dir, cfg):
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    ks = [f"recall@{k}" for k in cfg["RECALL_KS"]] + \
         [f"mrr@{cfg['MRR_K']}", f"hitrate@{cfg['HITRATE_K']}"]
    lines = [
        "# BPR-MF Baseline Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "Kernel: lb-bpr",
        "",
        "## Metrics",
        "",
        "| Model | Split | " + " | ".join(ks) + " | n |",
        "|---|---|" + "---|" * (len(ks) + 1),
    ]
    for model, splits in metrics.items():
        for split, m in splits.items():
            row = [model, split] + [f"{m[k]:.6f}" for k in ks] + [str(m["n_users"])]
            lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "## Architecture",
        "",
        f"- BPR-MF: user/item factor matrices, D={cfg['D']}",
        f"- Pairwise sampled loss: -log sigmoid(pos - neg)",
        f"- Optimizer: Adam lr={cfg['LR']}, epochs={cfg['EPOCHS']}",
        f"- Per-user positive cap per epoch: {cfg['PER_USER_CAP']}",
        f"- Negatives: uniform, up to {cfg['NEG_RETRIES']} resamples if in train set",
        "",
        "## Notes",
        "",
        "- Eval is set-based (not next-item): the full test set is ranked,",
        "  primary comparison metric is discovery recall.",
        "- Repeats are NOT excluded from candidate lists.",
        "- Same cohort seed and protocol as lb_ranker / lb_item2vec;",
        "  their numbers come from their own kernel reports.",
        "- Popularity and user-frequency baselines re-evaluated here for",
        "  direct comparability.",
    ]
    (reports_dir / "bpr_report.md").write_text("\n".join(lines) + "\n")

    readme = [
        "# lb-bpr outputs",
        "",
        "| File | Description |",
        "|---|---|",
        "| models/bpr_final.pt | BPR-MF factors (state_dict) |",
        "| reports/bpr_report.md | Metrics + architecture report |",
        "| metrics.json | Machine-readable metrics |",
        "",
        "## Consuming the model",
        "",
        "```python",
        "import torch",
        "sd = torch.load('models/bpr_final.pt', weights_only=True)",
        "user_factors = sd['user_factors.weight']   # (n_users, 64)",
        "item_factors = sd['item_factors.weight']   # (n_items, 64)",
        "scores = item_factors @ user_factors[u_idx]",
        "```",
        "",
        "Rerun: attach the lb-trainprep dataset and run this kernel.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n")
    log("reports written")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_bpr(events_path, vocab_path, output_dir, cfg=None):
    cfg = cfg or CONFIG
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_items = load_n_items(vocab_path)
    offsets, data, user_ids, user_index, train_counts, user_test = \
        load_csr(events_path, n_items)
    n_users = len(user_ids)

    log(f"n_users={n_users} (expected {EXPECTED['n_users']})")
    log(f"n_items={n_items} (expected {EXPECTED['n_items']})")
    log(f"train_rows={len(data)} (upper bound {EXPECTED['train_rows_upper']})")

    model, device = train_bpr(offsets, data, n_users, n_items, output_dir, cfg)
    metrics = evaluate(model, device, offsets, data, user_index, train_counts,
                       user_test, n_items, cfg)
    _write_reports(metrics, output_dir, cfg)
    log("DONE")
    print(json.dumps(metrics, indent=2))
    return metrics


# ---------------------------------------------------------------------------
# Local test
# ---------------------------------------------------------------------------

def _run_test():
    import tempfile

    import pyarrow as pa

    tmp = Path(tempfile.mkdtemp(prefix="bpr_test_"))
    rng = np.random.default_rng(0)

    n_users, n_items = 20, 50
    rows = {"user": [], "item_id": [], "split": []}
    for u in range(n_users):
        items = rng.choice(n_items, size=30, replace=True)
        for k, it in enumerate(items):
            rows["user"].append(u)
            rows["item_id"].append(int(it))
            rows["split"].append(0 if k < 25 else 2)
    pq.write_table(pa.table(rows), tmp / "events.parquet")
    pq.write_table(pa.table({"item_id": np.arange(n_items, dtype=np.int64)}),
                    tmp / "vocab.parquet")

    cfg = dict(CONFIG)
    cfg.update(EPOCHS=1, EVAL_SAMPLE=100, PER_USER_CAP=20, BATCH_TRIPLES=64)
    out = tmp / "out"
    metrics = run_bpr(tmp / "events.parquet", tmp / "vocab.parquet", out, cfg)

    assert "bpr_mf" in metrics and "overall" in metrics["bpr_mf"], "missing bpr metrics"
    assert (out / "models" / "bpr_final.pt").exists(), "missing model"
    assert (out / "reports" / "bpr_report.md").exists(), "missing report"
    assert (out / "metrics.json").exists(), "missing metrics.json"
    assert (out / "README.md").exists(), "missing README"
    print("ALL BPR TESTS PASSED")


# ---------------------------------------------------------------------------
# Kaggle entry
# ---------------------------------------------------------------------------

def main():
    root = Path("/kaggle/input")
    events = sorted(root.rglob("events.parquet"))
    vocabs = sorted(root.rglob("vocab.parquet"))
    log(f"inputs: events={len(events)} vocab={len(vocabs)}")
    for p in root.rglob("*"):
        if p.is_file():
            log(f"  {p}")
    if len(events) != 1 or len(vocabs) != 1:
        log(f"FATAL: need exactly one events.parquet and one vocab.parquet "
            f"(got {len(events)}/{len(vocabs)})")
        sys.exit(1)
    run_bpr(events[0], vocabs[0], Path("/kaggle/working"))


if __name__ == "__main__":
    if "--test" in sys.argv:
        _run_test()
    elif Path("/kaggle/input").exists():
        main()
    else:
        print("No /kaggle/input and no --test flag; nothing to do.")
