"""
Product Quantization of item2vec embeddings for on-device tiers.

Two tiers:
  SONATA: m=16 subspaces (dim 4 each) -> 16 bytes/row codes
  ETUDE:  m=8  subspaces (dim 8 each) ->  8 bytes/row codes

K-means PQ training (k=256) on a 200k random sample (seed 42), then
encode all items. Reconstruction quality evaluated via recall@100
against exact cosine top-100 (chunked to stay within 13 GB).

Input:  item2vec_final.npy (n x 64 float32), v1_vocab.parquet
Output: models/{sonata,etude}_pq.npz, reports/quantize_report.md, summary.json

CPU-only, numpy + stdlib (+ pyarrow for vocab parquet read).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ── CONFIG ─────────────────────────────────────────────────────
TRAIN_SAMPLE  = 200_000
KMEANS_ITERS  = 15
EVAL_Q        = 1000
EVAL_TOP      = 100
CHUNK_SIZE    = 100          # queries per matmul chunk
K             = 256
SEED          = 42

# Tier definitions: (name, m, subspace_dim)
TIERS = [
    ("sonata", 16, 4),
    ("etude",   8, 8),
]

T0 = time.time()
def log(m): print(f"[{int(time.time()-T0):>5}s] {m}", flush=True)

# ── discovery ──────────────────────────────────────────────────
def find_input(root: Path):
    """Locate embedding + vocab files under root."""
    emb_files = list(root.rglob("item2vec_final.npy"))
    if not emb_files:
        sys.exit("FATAL: item2vec_final.npy not found")
    if len(emb_files) > 1:
        sys.exit(f"FATAL: multiple item2vec_final.npy: {emb_files}")

    vocab_files = list(root.rglob("v1_vocab.parquet"))
    if not vocab_files:
        vocab_files = list(root.rglob("vocab.parquet"))
        if not vocab_files:
            sys.exit("FATAL: vocab parquet not found")
        if len(vocab_files) > 1:
            sys.exit(f"FATAL: multiple vocab parquet: {vocab_files}")
        log("WARNING: v1_vocab.parquet not found, using vocab.parquet")
    elif len(vocab_files) > 1:
        sys.exit(f"FATAL: multiple v1_vocab.parquet: {vocab_files}")

    return emb_files[0], vocab_files[0]

# ── load + normalize ───────────────────────────────────────────
def load_embeddings(path: Path):
    emb = np.load(str(path))
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb /= (norms + 1e-8)
    log(f"loaded embeddings {emb.shape}, dtype={emb.dtype}")
    return emb

# ── k-means (Lloyd) ───────────────────────────────────────────
def kmeans(X: np.ndarray, k: int, n_iter: int, rng: np.random.Generator):
    """Lloyd k-means. X: (n, d). Returns centroids (k, d) as float32."""
    n, d = X.shape
    # init: k distinct random rows
    idxs = rng.choice(n, size=k, replace=False)
    centroids = X[idxs].copy().astype(np.float32)

    for it in range(n_iter):
        # assignment: ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2
        Xf = X.astype(np.float32)
        x_norm2 = (Xf * Xf).sum(axis=1)                    # (n,)
        c_norm2 = (centroids * centroids).sum(axis=1)       # (k,)
        dist = x_norm2[:, None] - 2.0 * Xf @ centroids.T + c_norm2[None, :]  # (n, k)
        labels = dist.argmin(axis=1)                        # (n,)

        # update centroids (mean of assigned points)
        new_c = np.zeros_like(centroids)
        counts = np.zeros(k, dtype=np.float32)
        for j in range(k):
            mask = labels == j
            if mask.any():
                new_c[j] = Xf[mask].mean(axis=0)
                counts[j] = mask.sum()
            else:
                # empty cluster -> reinit from random data row
                new_c[j] = Xf[rng.integers(n)]
        centroids = new_c

    log(f"kmeans done: k={k}, iters={n_iter}")
    return centroids

# ── PQ train + encode ─────────────────────────────────────────
def pq_train_encode(emb: np.ndarray, m: int, subspace_dim: int,
                    sample_rows: np.ndarray):
    """
    Train PQ codebooks on sample_rows, then encode all rows.
    Returns codes (n, m) uint8, codebooks (m, 256, subspace_dim) float32.
    """
    rng = np.random.default_rng(SEED)
    n, d = emb.shape
    codebooks = np.zeros((m, K, subspace_dim), dtype=np.float32)
    codes = np.zeros((n, m), dtype=np.uint8)

    for j in range(m):
        lo = j * subspace_dim
        hi = lo + subspace_dim
        Xj = sample_rows[:, lo:hi]          # (TRAIN_SAMPLE, subspace_dim)
        centroids = kmeans(Xj, K, KMEANS_ITERS, rng)  # (K, subspace_dim)
        codebooks[j] = centroids

        # encode all rows: find nearest centroid per subspace
        # chunk to keep memory bounded
        chunk = 500_000
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            Xj_all = emb[s:e, lo:hi].astype(np.float32)
            # dist = ||x||^2 - 2 x.c + ||c||^2
            x_n2 = (Xj_all * Xj_all).sum(axis=1)
            c_n2 = (centroids * centroids).sum(axis=1)
            dmat = x_n2[:, None] - 2.0 * Xj_all @ centroids.T + c_n2[None, :]
            codes[s:e, j] = dmat.argmin(axis=1).astype(np.uint8)

    log(f"PQ encode done: codes {codes.shape}, codebooks {codebooks.shape}")
    return codes, codebooks

# ── reconstruct from codes ────────────────────────────────────
def reconstruct(codes: np.ndarray, codebooks: np.ndarray):
    """(n, m) codes + (m, 256, subspace_dim) codebooks -> (n, d) float32."""
    n, m = codes.shape
    sub_dim = codebooks.shape[2]
    d = m * sub_dim
    rec = np.zeros((n, d), dtype=np.float32)
    for j in range(m):
        rec[:, j * sub_dim : (j + 1) * sub_dim] = codebooks[j, codes[:, j]]
    return rec

# ── recall@K eval ─────────────────────────────────────────────
def recall_at_k(exact_top, approx_top):
    """exact_top, approx_top: (q, k) index arrays. Return mean recall."""
    q = exact_top.shape[0]
    hits = np.zeros(q, dtype=np.float32)
    exact_sets = [set(exact_top[i].tolist()) for i in range(q)]
    for i in range(q):
        hits[i] = len(exact_sets[i] & set(approx_top[i].tolist()))
    return float(hits.mean() / exact_top.shape[1])

# ── exact cosine top-K (chunked) ──────────────────────────────
def exact_topk(emb: np.ndarray, q_idxs: np.ndarray, k: int):
    """Row-normalized emb, query indices -> (q, k) exact top-k indices."""
    q = len(q_idxs)
    results = np.zeros((q, k), dtype=np.int64)
    for s in range(0, q, CHUNK_SIZE):
        e = min(s + CHUNK_SIZE, q)
        sims = emb[q_idxs[s:e]] @ emb.T          # (chunk, n)
        # argpartition top-k
        ks = min(k, sims.shape[1] - 1)
        part = np.argpartition(-sims, ks, axis=1)[:, :k]
        # sort within top-k
        rows = np.arange(e - s)[:, None]
        order = np.argsort(-sims[rows, part], axis=1)
        results[s:e] = part[rows, order]
    return results

# ── main PQ pipeline ──────────────────────────────────────────
def run_pq(emb: np.ndarray, out_dir: Path):
    rng = np.random.default_rng(SEED)
    n, d = emb.shape

    # training sample (cap to available rows for small test sets)
    n_sample = min(TRAIN_SAMPLE, n)
    sample_idxs = rng.choice(n, size=n_sample, replace=False)
    sample_rows = emb[sample_idxs].copy()
    log(f"sampled {n_sample} rows for k-means training")

    tier_results = {}
    for name, m, sub_dim in TIERS:
        log(f"--- tier {name}: m={m}, sub_dim={sub_dim} ---")
        t1 = time.time()
        codes, codebooks = pq_train_encode(emb, m, sub_dim, sample_rows)

        # reconstruct for recall eval
        rec = reconstruct(codes, codebooks)
        mse = float(np.mean((emb - rec) ** 2))
        log(f"  reconstruction MSE = {mse:.6f}")

        # eval: recall@100
        eval_idxs = rng.choice(n, size=EVAL_Q, replace=False)
        exact = exact_topk(emb, eval_idxs, EVAL_TOP)
        # approx: cosine top-100 on reconstructed vectors
        approx = exact_topk(rec, eval_idxs, EVAL_TOP)
        recall = recall_at_k(exact, approx)
        log(f"  recall@{EVAL_TOP} = {recall:.4f}")

        # storage
        code_bytes = codes.nbytes
        cb_bytes = codebooks.nbytes
        total_bytes = code_bytes + cb_bytes
        f32_bytes = emb.nbytes
        ratio = f32_bytes / total_bytes

        tier_results[name] = dict(
            m=m, subspace_dim=sub_dim,
            code_shape=list(codes.shape),
            codebook_shape=list(codebooks.shape),
            code_bytes=code_bytes, codebook_bytes=cb_bytes,
            total_bytes=total_bytes,
            f32_bytes=f32_bytes, compression_ratio=round(ratio, 2),
            reconstruction_mse=round(mse, 8),
            recall_at_k=round(recall, 6),
        )
        log(f"  storage: codes {code_bytes:,} + codebooks {cb_bytes:,} = {total_bytes:,} bytes")
        log(f"  compression: {ratio:.2f}x vs f32 ({f32_bytes:,} bytes)")

        # save
        models_dir = out_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(models_dir / f"{name}_pq.npz"),
            codes=codes, codebooks=codebooks, centroids=codebooks,
            n_items=np.array([n]), d=np.array([d]),
            m=np.array([m]),
        )
        log(f"  saved {models_dir / f'{name}_pq.npz'}")

    # ── report ─────────────────────────────────────────────────
    rep_dir = out_dir / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    f32_bytes = emb.nbytes
    lines = [
        "# Product Quantization Report", "",
        "PQ compresses item2vec embeddings for on-device ANN serving.", "",
        "## Tier comparison", "",
        "| Tier | Subspaces | Bytes/row | Total storage | Compression | Recon MSE | Recall@100 |",
        "|------|-----------|-----------|---------------|-------------|-----------|------------|",
    ]
    for name, m, sub_dim in TIERS:
        r = tier_results[name]
        bpr = r["total_bytes"] // n
        lines.append(
            f"| {name.capitalize()} | {r['m']} | {bpr} | "
            f"{r['total_bytes']:,} B | {r['compression_ratio']:.1f}x | "
            f"{r['reconstruction_mse']:.6f} | {r['recall_at_k']:.4f} |"
        )
    lines += [
        "", f"Baseline f32 storage: {f32_bytes:,} bytes ({f32_bytes / 1e6:.1f} MB).",
        f"Items: {n:,}. Embedding dim: {d}.",
        "", "PQ is lossy: reconstruction MSE quantifies vector distortion.",
        "Recall@100 measures ANN quality (fraction of exact top-100 found in PQ top-100).",
        "", "## Notes", "",
        "- k-means k=256, 15 Lloyd iterations, trained on 200k random sample (seed 42).",
        "- SONATA (16 subspaces x dim 4) offers finer granularity.",
        "- ETUDE (8 subspaces x dim 8) is half the storage, coarser granularity.",
        "- On-device tiering: ETUDE for mobile/embedded, SONATA for high-end.",
    ]
    (rep_dir / "quantize_report.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {rep_dir / 'quantize_report.md'}")

    # summary
    summary = dict(
        n_items=n, d=d, tiers=tier_results,
        train_sample=TRAIN_SAMPLE, kmeans_iters=KMEANS_ITERS,
        eval_queries=EVAL_Q, eval_top_k=EVAL_TOP,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log("DONE")
    print(json.dumps(summary))
    return summary

# ── synthetic test (no pyarrow) ───────────────────────────────
def run_test():
    """Fast local test with fake data. No pyarrow, no parquet."""
    import tempfile, os
    rng = np.random.default_rng(SEED)
    n, d = 5_000, 64
    emb = rng.standard_normal((n, d)).astype(np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        log(f"[TEST] synthetic {n}x{d} embeddings")
        result = run_pq(emb, out)

        # sanity checks
        for name in ["sonata", "etude"]:
            f = out / "models" / f"{name}_pq.npz"
            assert f.exists(), f"missing {f}"
            with np.load(str(f)) as data:
                codes = data["codes"]
                cb = data["codebooks"]
                assert codes.shape[0] == n
                assert codes.min() >= 0 and codes.max() < K
                assert cb.shape[0] == codes.shape[1]

        for name in ["sonata", "etude"]:
            r = result["tiers"][name]
            assert 0.0 <= r["recall_at_k"] <= 1.0, f"recall out of range: {r}"
            log(f"[TEST] {name}: recall@{EVAL_TOP}={r['recall_at_k']:.4f}, "
                f"MSE={r['reconstruction_mse']:.6f}, "
                f"storage={r['total_bytes']:,} bytes")

        log("[TEST] all checks passed")

# ── entry point ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Run synthetic test (no pyarrow, no parquet)")
    args = parser.parse_args()

    if args.test:
        run_test()
        return

    root = Path("/kaggle/input")
    emb_path, vocab_path = find_input(root)
    log(f"Embeddings: {emb_path}")
    log(f"Vocab: {vocab_path}")

    emb = load_embeddings(emb_path)

    # load vocab to assert row count match
    import pyarrow.parquet as pq
    vt = pq.read_table(str(vocab_path))
    n_vocab = vt.num_rows
    log(f"vocab rows: {n_vocab}")
    assert emb.shape[0] == n_vocab, \
        f"emb rows ({emb.shape[0]}) != vocab rows ({n_vocab})"

    out_dir = Path("/kaggle/working")
    run_pq(emb, out_dir)


if __name__ == "__main__":
    main()
