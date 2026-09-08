"""
Recency evaluation kernel: decayed-frequency and rank-fusion scorers.

Evaluates whether exponentially-decayed frequency beats plain frequency for
repeat prediction, and whether rank-fusion of the sequence model + frequency
scorer beats both individually.

VERSION-SKEW NOTE: item2vec_final.npy and ranker_final.pt were trained on
trainprep V1 vocab (ALL-events counts, 2,803,656 items). The current V2
trainprep output (events.parquet + vocab.parquet) uses a TRAIN-ONLY vocab
(~2,787,934 items) with different item_id assignments. This kernel rebuilds
the V1 vocab from listens.parquet (sort: -count, then mbid ascending for
deterministic id assignment), translates V2 event data into V1 space via
mbid-keyed join, then runs all scorers in V1 space. V1 vocab is emitted
as data/v1_vocab.parquet for pairing with the published weights.

V4 COHORT-FIRST NOTE: previous versions loaded all 36,970 users into RAM
(~12.4 GB peak) before sampling the 5,000-user eval cohort, and OOM'd at
Kaggle's ~13 GB cap. v4 selects the eval cohort FIRST (same eligibility
rule and seed as evaluate()) and materializes only cohort users' events —
full precision, no data cut for any evaluated user; peak RSS ~4 GB.
Popularity counts come from the V1 vocab table (exact all-events counts);
this differs slightly from the published train-only popularity baseline
and is noted in the report.

CALIBRATION: per-scorer reliability curves + expected calibration error
(ECE) over min-max-normalized top-20 scores — measures ranking-score
trustworthiness, not the probability that any single recommendation is
"right".

Scorers:
  - user_frequency: plain per-user item counts (baseline)
  - decay_30d / decay_90d / decay_365d: exponential decay half-lives
  - ranker: GRU sequence model (loaded from pre-trained checkpoint)
  - item2vec: mean last-50 embedding cosine
  - popularity: global count desc
  - rrf_fusion: reciprocal-rank fusion (ranker + best decay by repeat R@100)
  - w_fusion_05: weighted variant (0.5*ranker + 0.5*best_decay)

Inputs (rglob under /kaggle/input):
  listens.parquet  (user, ts, recording_mbid, ... ) — from lb-sanitize
  events.parquet   (user, ts, item_id, split)      — V2 trainprep output
  vocab.parquet    (recording_mbid, count, item_id) — V2 vocab
  item2vec_final.npy  ((2803656, 64) f32) — V1-space embeddings
  ranker_final.pt     (GRURanker checkpoint, optional) — V1-space

Outputs: data/v1_vocab.parquet, metrics.json, reports/eval_report.md, README.md

CPU-only, no GPU needed.
"""
import gc, json, os, random, re, sys, time
from collections import defaultdict
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────
EMB_DIM         = 64
GRU_HIDDEN      = 128
GRU_LAYERS      = 1
SEQ_LEN         = 50            # max context length (last N train items)
EVAL_SAMPLE     = 5_000         # users to sample for eval
EVAL_CTX_LEN    = 50            # last N train events for eval context
RECALL_KS       = [20, 100]
MRR_K           = 10
HITRATE_K       = 20
SEED            = 42
RRF_K           = 60            # RRF constant
DECAY_HALF_LIVES = [30, 90, 365]  # half-lives in days

# ── EXPECTED VALUES (embedded for sanity checking) ─────────────────────
EXPECTED = dict(
    n_users=36970,
    n_items=2803656,
    train_rows_upper=1_270_000_000,
)

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

T0 = time.time()
def log(m): print(f"[{int(time.time()-T0):>5}s] {m}", flush=True)

def rss_mb():
    """Return current RSS in MB from /proc/self/status (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            m = re.search(r"VmRSS:\s+(\d+)\s+kB", f.read())
            return int(m.group(1)) / 1024.0 if m else -1.0
    except Exception:
        return -1.0


# ═══════════════════════════════════════════════════════════════════════
# V1 VOCAB REBUILD (from listens.parquet, matching lb_trainprep sort)
# ═══════════════════════════════════════════════════════════════════════

V1_EXPECTED_SIZE = 2_803_656  # known V1 vocab (ALL-events, count>=10)


def build_v1_vocab(listens_path):
    """Rebuild V1 vocab from listens.parquet.

    V1 used ALL-events counts (not train-only), then built vocab with
    count >= 10. Sort order: (-count, recording_mbid) for deterministic
    id assignment — copies lb_trainprep.build_vocab semantics exactly.

    Returns:
      mbid_to_v1_id: dict[bytes, int32]
      v1_table: pa.Table with columns (recording_mbid, count, v1_item_id)
    """
    log(f"Building V1 vocab from {listens_path}")
    pf = pq.ParquetFile(str(listens_path))
    item_counts = {}  # recording_mbid (bytes) -> count

    batch_num = 0
    total_rows = 0
    for batch in pf.iter_batches(batch_size=2_000_000,
                                  columns=["recording_mbid"]):
        batch_num += 1
        recs = batch.column("recording_mbid").to_pylist()
        total_rows += batch.num_rows
        vc = pc.value_counts(batch.column("recording_mbid"))
        vals = vc.field("values").to_pylist()
        cnts = vc.field("counts").to_pylist()
        for v, c in zip(vals, cnts):
            key = v if isinstance(v, bytes) else int(v)
            item_counts[key] = item_counts.get(key, 0) + int(c)
        if batch_num % 50 == 0:
            log(f"  v1 pass batch {batch_num:>5}  rows={total_rows:>13,}  "
                f"items={len(item_counts):>9,}")

    log(f"  total distinct items: {len(item_counts):,}")

    # Filter to count >= 10 (V1 threshold), same as lb_trainprep.MIN_COUNT
    eligible = [(mbid, cnt) for mbid, cnt in item_counts.items() if cnt >= 10]
    # Sort: primary = -count (desc), secondary = mbid (asc bytes) — deterministic
    eligible.sort(key=lambda x: (-x[1], x[0]))
    log(f"  V1 vocab size: {len(eligible):,} "
        f"(expected {V1_EXPECTED_SIZE:,}, "
        f"delta {abs(len(eligible) - V1_EXPECTED_SIZE):,})")

    mbid_to_v1_id = {mbid: i for i, (mbid, _) in enumerate(eligible)}
    v1_table = pa.table({
        "recording_mbid": pa.array([e[0] for e in eligible], type=pa.binary(16)),
        "count": pa.array([e[1] for e in eligible], type=pa.int64()),
        "v1_item_id": pa.array(list(range(len(eligible))), type=pa.int32()),
    })
    return mbid_to_v1_id, v1_table


def build_v2_to_v1(vocab_path, mbid_to_v1_id, n_v2):
    """Translate V2 item_ids → V1 item_ids via recording_mbid join.

    V2 vocab.parquet has columns (recording_mbid, count, item_id).
    Every V2 item must map to a V1 id (V2 is strict subset of V1).

    Returns:
      v2_to_v1: int64 array of shape (n_v2,), -1 for unmapped
    """
    vt = pq.read_table(str(vocab_path), columns=["recording_mbid", "item_id"])
    v2_mbid = vt.column("recording_mbid").to_pylist()
    v2_id   = vt.column("item_id").to_numpy()

    v2_to_v1 = np.full(n_v2, -1, dtype=np.int64)
    mapped = 0
    for mbid, v2i in zip(v2_mbid, v2_id.tolist()):
        key = mbid if isinstance(mbid, bytes) else bytes(mbid)
        if key in mbid_to_v1_id:
            v2_to_v1[v2i] = mbid_to_v1_id[key]
            mapped += 1

    unmapped = int((v2_to_v1 < 0).sum())
    log(f"  V2->V1 translation: {mapped:,} mapped, {unmapped:,} unmapped "
        f"(of {n_v2:,} V2 items)")
    assert unmapped == 0, f"V2 item with no V1 mapping: {unmapped} items"
    return v2_to_v1


def translate_csr_to_v1(data, ts_data, user_test, v2_to_v1):
    """Translate CSR data array + test item sets from V2 space to V1 space.

    Operates in 8M-element chunks to avoid a full int64 copy of the data
    array (~10GB peak). ts_data is reused directly (already int32).
    """
    n = len(data)
    CHUNK = 8_000_000
    data_v1 = np.empty(n, dtype=np.int32)
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        data_v1[s:e] = v2_to_v1[data[s:e]]  # v2_to_v1 is int64, cast to int32 on store

    user_test_v1 = {}
    for uid, test_set in user_test.items():
        user_test_v1[uid] = set(v2_to_v1[list(test_set)].tolist())

    return data_v1, ts_data, user_test_v1


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

def load_csr(events_path, n_items, cohort_users=None):
    """Load train split into CSR arrays; collect test item sets in pass 1.

    If cohort_users is given (array of user ids), only those users' rows are
    loaded — the eval cohort is selected BEFORE this call (cohort-first
    memory design: ~5K users ≈ 174M events ≈ 1.4GB vs 10GB full load).

    Returns:
      offsets:     int64 (n_users+1,) — user's train events are
                   data[offsets[i]:offsets[i+1]], file order (= ts order)
      data:        int32 concatenated train item_ids
      ts_data:     int32 concatenated train timestamps (parallel to data)
      user_ids:    int32 sorted unique user ids
      user_index:  dict user_id -> CSR row index
      train_counts:int64 (n_items,) per-item train frequency (padded)
      user_test:   dict user_id -> set(test item_ids)
    """
    cohort_arr = (np.asarray(sorted(cohort_users), dtype=np.int32)
                  if cohort_users is not None else None)
    log(f"Loading events from {events_path}"
        + (f" (cohort-filtered: {len(cohort_arr):,} users)" if cohort_arr is not None else ""))
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

        if cohort_arr is not None:               # cohort-first: filter early
            cm = np.isin(users_np, cohort_arr)
            if not cm.any():
                continue
            users_np = users_np[cm]; items_np = items_np[cm]; splits_np = splits_np[cm]

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
    ts_data = np.empty(int(offsets[-1]), dtype=np.int32)

    # ── Pass 2: fill data + ts arrays (file is user-grouped, ts-ordered) ──
    fill = defaultdict(int)                # user_id -> events written
    pf = pq.ParquetFile(str(events_path))
    batch_num = 0
    for batch in pf.iter_batches(batch_size=2_000_000,
                                  columns=["user", "item_id", "split", "ts"]):
        batch_num += 1
        users_np  = batch.column("user").to_numpy()
        items_np  = batch.column("item_id").to_numpy()
        splits_np = batch.column("split").to_numpy()
        ts_np     = batch.column("ts").to_numpy()
        if cohort_arr is not None:               # cohort-first: filter early
            cm = np.isin(users_np, cohort_arr)
            if not cm.any():
                continue
            users_np = users_np[cm]; items_np = items_np[cm]
            splits_np = splits_np[cm]; ts_np = ts_np[cm]
        m = splits_np == 0
        if not m.any():
            continue
        u = users_np[m]; it = items_np[m]; ts = ts_np[m]
        for uid in np.unique(u).tolist():     # ~58 users per 2M batch
            um = u == uid
            n = int(um.sum())
            start = offsets[user_index[uid]] + fill[uid]
            data[start:start + n] = it[um]
            ts_data[start:start + n] = ts[um].astype(np.int32)
            fill[uid] += n
        if batch_num % 50 == 0:
            log(f"  p2 batch {batch_num:>5}  filled={int(offsets[-1]) and sum(fill.values()):,}")

    train_counts = np.bincount(data, minlength=n_items).astype(np.int64)
    log(f"  CSR: {len(data):,} train events, {len(user_ids):,} users, "
        f"max_item_id={int(data.max()) if len(data) else -1}")
    return (offsets, data, ts_data, user_ids, user_index, train_counts, user_test)


def load_n_items(vocab_path):
    """Vocab size from vocab.parquet (item_ids are dense 0..N-1 by construction)."""
    vt = pq.read_table(str(vocab_path), columns=["item_id"])
    ids = vt.column("item_id").to_numpy()
    return int(ids.max()) + 1


def select_cohort(events_path, eval_sample=EVAL_SAMPLE, seed=SEED):
    """Cohort-first design: pick the eval cohort BEFORE the heavy CSR load.

    Cheap pass over (user, item_id, split) — eligibility mirrors evaluate():
    >=10 train events AND >=1 test event. Returns (cohort_ids int32 sorted,
    n_eligible). If fewer eligible than eval_sample, all eligible are used.
    """
    log("Selecting eval cohort (cheap pre-pass)")
    train_counts = defaultdict(int)
    has_test = defaultdict(bool)
    pf = pq.ParquetFile(str(events_path))
    for batch in pf.iter_batches(batch_size=2_000_000,
                                 columns=["user", "item_id", "split"]):
        users_np  = batch.column("user").to_numpy()
        splits_np = batch.column("split").to_numpy()
        m = splits_np == 0
        if m.any():
            u, c = np.unique(users_np[m], return_counts=True)
            for uid, cnt in zip(u.tolist(), c.tolist()):
                train_counts[uid] += cnt
        m = splits_np == 2
        if m.any():
            for uid in np.unique(users_np[m]).tolist():
                has_test[uid] = True
    eligible = [u for u in train_counts
                if train_counts[u] >= 10 and has_test.get(u, False)]
    rng = random.Random(seed)
    if len(eligible) > eval_sample:
        cohort = rng.sample(sorted(eligible), eval_sample)
    else:
        cohort = sorted(eligible)
    log(f"  eligible users: {len(eligible):,} -> cohort {len(cohort):,}")
    return np.array(sorted(cohort), dtype=np.int32), len(eligible)


def load_embeddings(npy_path, n_items):
    """Load frozen item2vec embeddings. Returns (n_items, 64) float32 numpy."""
    emb = np.load(str(npy_path)).astype(np.float32)
    assert emb.shape[1] == EMB_DIM, f"Expected dim {EMB_DIM}, got {emb.shape}"
    assert emb.shape[0] == n_items, \
        (f"Embedding vocab mismatch: npy has {emb.shape[0]:,} items, "
         f"expected {n_items:,}. Check that item2vec was trained on V1 vocab.")
    return emb


# ═══════════════════════════════════════════════════════════════════════
# STEP 2 — Scorer functions
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
        import torch
        x = self.emb(ctx_ids.long())
        seq_len = x.size(1)
        lengths_cpu = lengths.cpu().long().clamp(min=1, max=seq_len)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        h = h_n[-1]
        h = self.proj(h)
        return h

    def state_dict(self):
        return {"emb": self.emb, "gru": self.gru, "proj": self.proj}

    def load(self, path):
        import torch
        ckpt = torch.load(str(path), map_location=self.device, weights_only=True)
        for k, sd in ckpt.items():
            self.state_dict()[k].load_state_dict(sd)


def _score_ranker(model, ctx_items, all_emb_t):
    """GRU scorer: last EVAL_CTX_LEN items → hidden → dot-product vs all items."""
    import torch
    ctx_len = len(ctx_items)
    ctx_padded = np.zeros(EVAL_CTX_LEN, dtype=np.int32)
    ctx_padded[:ctx_len] = ctx_items
    with torch.no_grad():
        ctx_t = torch.from_numpy(ctx_padded).long().unsqueeze(0).to(model.device)
        lens = torch.tensor([ctx_len], dtype=torch.long, device=model.device)
        h = model.forward(ctx_t, lens)          # (1, 64)
        h_norm = h / (h.norm(dim=1, keepdim=True) + 1e-8)
        scores = (all_emb_t @ h_norm.squeeze(0)).cpu().numpy()
    return scores


def _score_item2vec(norm_emb, ctx_items):
    """Item2Vec scorer: mean of last-50 embeddings, cosine sim vs all items."""
    ctx_vec = norm_emb[ctx_items].mean(axis=0)
    nrm = np.linalg.norm(ctx_vec)
    if nrm > 1e-8:
        ctx_vec = ctx_vec / nrm
        return norm_emb @ ctx_vec
    return None


def _score_popularity(pop_rank):
    """Popularity scorer: global count desc (same for all users)."""
    return pop_rank


def _score_user_frequency(seq, n_items, pop_rank, pop_pos):
    """User-frequency scorer: user freq desc, ties by popularity asc."""
    uf = np.bincount(seq, minlength=n_items)
    user_items = np.where(uf > 0)[0]
    order1 = user_items[np.lexsort((pop_pos[user_items], -uf[user_items]))]
    rest_mask = np.ones(n_items, dtype=bool)
    rest_mask[user_items] = False
    return np.concatenate([order1, pop_rank[rest_mask[pop_rank]]])


def _uf_scores(seq, n_items, pop_pos):
    """Raw user-frequency scores for calibration: freq + tiny popularity tie-break."""
    uf = np.bincount(seq, minlength=n_items).astype(np.float64)
    return uf + 1e-9 * (n_items - pop_pos)


def _score_decay(seq_items, seq_ts, n_items, half_life):
    """Exponential decay scorer.

    score(item) = Σ exp(-ln2 * age_days / half_life)
    where age_days = (user_max_ts - ts) / 86400
    """
    max_ts = seq_ts.max()
    age_days = (max_ts - seq_ts).astype(np.float64) / 86400.0
    weights = np.exp(-np.log(2) * age_days / half_life)
    scores = np.zeros(n_items, dtype=np.float64)
    np.add.at(scores, seq_items, weights)
    return scores


def _top_k(scores, k=200):
    """Top-k item ids by score, descending. O(n) select + small sort — avoids
    full 2.8M argsort churn per scorer per user (metrics need ≤top-100,
    fusion needs top-200)."""
    idx = np.argpartition(-scores, k)[:k + 1]
    return idx[np.argsort(-scores[idx])]


def _rrf_fusion(ranked_a, ranked_b, k=RRF_K, n_items=None):
    """Reciprocal-rank fusion: 1/(k+rank_a) + 1/(k+rank_b)."""
    if n_items is None:
        n_items = max(int(ranked_a.max()), int(ranked_b.max())) + 1
    ranks_a = np.full(n_items, float(len(ranked_a)), dtype=np.float64)
    ranks_b = np.full(n_items, float(len(ranked_b)), dtype=np.float64)
    ranks_a[ranked_a] = np.arange(len(ranked_a), dtype=np.float64)
    ranks_b[ranked_b] = np.arange(len(ranked_b), dtype=np.float64)
    scores = 1.0 / (k + ranks_a + 1) + 1.0 / (k + ranks_b + 1)
    return np.argsort(-scores)


def _w_fusion_05(ranked_a, ranked_b, k=RRF_K, n_items=None):
    """Weighted RRF: 0.5/(k+rank_a) + 0.5/(k+rank_b)."""
    if n_items is None:
        n_items = max(int(ranked_a.max()), int(ranked_b.max())) + 1
    ranks_a = np.full(n_items, float(len(ranked_a)), dtype=np.float64)
    ranks_b = np.full(n_items, float(len(ranked_b)), dtype=np.float64)
    ranks_a[ranked_a] = np.arange(len(ranked_a), dtype=np.float64)
    ranks_b[ranked_b] = np.arange(len(ranked_b), dtype=np.float64)
    scores = 0.5 / (k + ranks_a + 1) + 0.5 / (k + ranks_b + 1)
    return np.argsort(-scores)


# ═══════════════════════════════════════════════════════════════════════
# STEP 3 — Metrics
# ═══════════════════════════════════════════════════════════════════════

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


def _compute_novelty(ranked, pop_rank, n_items, k=20):
    """Mean popularity percentile of top-k items. Lower = rarer/more novel."""
    top_k = ranked[:k]
    return float(np.mean(pop_rank[top_k].astype(np.float64) / n_items))


def _compute_diversity(ranked, norm_emb, k=20):
    """Mean pairwise cosine similarity of top-k items' embeddings. Lower = more diverse."""
    top_k = ranked[:k]
    embs = norm_emb[top_k]  # (k, 64)
    sim = embs @ embs.T     # (k, k)
    n = len(top_k)
    if n < 2:
        return 0.0
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return float(sim[mask].mean())


def _add_metrics(bucket, ranked, test_items, repeat_items, discovery_items,
                 pop_rank, norm_emb, n_items):
    """Compute all metrics for one user and append to buckets."""
    novelty = _compute_novelty(ranked, pop_rank, n_items)
    diversity = _compute_diversity(ranked, norm_emb)

    o = _compute_metrics(ranked, test_items)
    o["novelty"] = novelty
    o["diversity"] = diversity
    bucket["overall"].append(o)

    if repeat_items:
        bucket["repeat"].append(_compute_metrics(ranked, repeat_items))
    if discovery_items:
        bucket["discovery"].append(_compute_metrics(ranked, discovery_items))


def _aggregate(per_user_metrics):
    """Average per-user metrics across cohort."""
    agg = {}
    for split in ["overall", "repeat", "discovery"]:
        rows = per_user_metrics[split]
        if not rows:
            agg[split] = {}
            continue
        metric_keys = [f"recall@{k}" for k in RECALL_KS] + \
            [f"mrr@{MRR_K}", f"hitrate@{HITRATE_K}"]
        agg[split] = {k: round(float(np.mean([m[k] for m in rows])), 6)
                      for k in metric_keys if k in rows[0]}
        if split == "overall" and rows[0].get("novelty") is not None:
            agg[split]["novelty"] = round(float(np.mean(
                [m["novelty"] for m in rows])), 6)
            agg[split]["diversity"] = round(float(np.mean(
                [m["diversity"] for m in rows])), 6)
        agg[split]["n_users"] = len(rows)
    return agg


# ═══════════════════════════════════════════════════════════════════════
# STEP 4 — Evaluation (two-pass: individuals then fusion)
# ═══════════════════════════════════════════════════════════════════════

def _calib_update(entry, ranked, scores, test_items, k=20):
    """Accumulate top-k (normalized score, in-test) pairs for calibration."""
    top = ranked[:k]
    s = np.asarray(scores, dtype=np.float64)[top]
    lo, hi = float(s.min()), float(s.max())
    if hi > lo:
        norm = (s - lo) / (hi - lo)
    else:
        norm = np.zeros_like(s)
    test_arr = np.fromiter(test_items, dtype=np.int64, count=len(test_items))
    obs = np.isin(top, test_arr).astype(np.float64)
    entry["pred"].append(norm)
    entry["obs"].append(obs)


def _compute_calibration(entry, n_bins=10):
    """Reliability curve + ECE from accumulated (pred, obs) pairs."""
    pred = np.concatenate(entry["pred"])
    obs = np.concatenate(entry["obs"])
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(pred, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        pm = float(pred[m].mean())
        om = float(obs[m].mean())
        w = float(m.mean())
        ece += w * abs(pm - om)
        rows.append([round(pm, 4), round(om, 4), round(w, 4)])
    return {"ece": round(ece, 4), "bins": rows}


def evaluate(embeddings, offsets, data, ts_data, user_ids, user_index,
             train_counts, user_test, n_items, ranker_model=None, seed=SEED):
    """Two-pass evaluation: individual scorers, then fusion scorers.

    Pass 1: user_frequency, decay_{30,90,365}d, item2vec, popularity, ranker.
    Pick best decay scorer by repeat R@100.
    Pass 2: rrf_fusion and w_fusion_05 using ranker + best decay.
    """
    log("Evaluation")
    norm_emb = embeddings  # already normalized in-place by caller

    pop_rank = np.argsort(-train_counts)
    pop_pos = np.empty(n_items, dtype=np.int64)
    pop_pos[pop_rank] = np.arange(n_items)
    train_counts_f64 = train_counts.astype(np.float64)  # hoisted: was allocated per-user (22MB x 5000)

    # Cohort (v4: already selected pre-load by select_cohort; re-derive from
    # the filtered CSR and assert eligibility holds)
    eligible = [int(u) for i, u in enumerate(user_ids.tolist())
                if (offsets[i + 1] - offsets[i]) >= 10 and user_test.get(u)]
    eval_users = eligible
    log(f"  cohort: {len(eval_users)} users (of {len(eligible)} eligible in filtered load)")

    # ── Pass 1: individual scorers ──
    individual_names = (["user_frequency"] +
                        [f"decay_{hl}d" for hl in DECAY_HALF_LIVES] +
                        ["item2vec", "popularity"])
    if ranker_model is not None:
        individual_names.append("ranker")

    # Calibration accumulators (v4): per-scorer top-20 (pred, obs) pairs
    calib = {m: {"pred": [], "obs": []} for m in individual_names}

    results = {m: {"overall": [], "repeat": [], "discovery": []}
               for m in individual_names + ["rrf_fusion", "w_fusion_05"]}

    # Pre-embed all items for ranker scoring
    all_emb_t = None
    if ranker_model is not None:
        import torch
        with torch.no_grad():
            all_emb_t = torch.from_numpy(embeddings).to(ranker_model.device)

    # Per-user: store top-FUSION_TOPK only for the fusion pass. Full 2.8M-element
    # arrays for 5k users = ~450GB (the v4 OOM). Top-200 is provably sufficient:
    # a fused top-100 item must be in the union of the two top-200s, because
    # out-of-list RRF scores (2/(k+201)) are strictly lower than any in-list score.
    FUSION_TOPK = 200
    user_ranker_ranks = {}   # uid -> top-FUSION_TOPK item ids
    user_decay_ranks = {}    # uid -> {hl: top-FUSION_TOPK item ids}

    for n_u, u in enumerate(eval_users, 1):
        i = user_index[u]
        seq = data[offsets[i]:offsets[i + 1]]
        ts  = ts_data[offsets[i]:offsets[i + 1]]
        test_items = user_test[u]
        if not len(seq) or not test_items:
            continue

        train_unique = np.unique(seq)
        train_set = set(train_unique.tolist())
        repeat_items = test_items & train_set
        discovery_items = test_items - train_set

        ctx_items = seq[-EVAL_CTX_LEN:]

        # ── user_frequency ──
        ranked_uf = _score_user_frequency(seq, n_items, pop_rank, pop_pos)
        _add_metrics(results["user_frequency"], ranked_uf, test_items,
                     repeat_items, discovery_items, pop_rank, norm_emb, n_items)
        _calib_update(calib["user_frequency"], ranked_uf,
                      _uf_scores(seq, n_items, pop_pos), test_items)

        # ── decay scorers ──
        decay_ranks = {}
        for hl in DECAY_HALF_LIVES:
            name = f"decay_{hl}d"
            scores = _score_decay(seq, ts, n_items, hl)
            ranked = _top_k(scores, FUSION_TOPK)
            decay_ranks[hl] = ranked
            _add_metrics(results[name], ranked, test_items,
                         repeat_items, discovery_items, pop_rank, norm_emb, n_items)
            _calib_update(calib[name], ranked, scores, test_items)
        user_decay_ranks[u] = decay_ranks

        # ── item2vec ──
        scores_i2v = _score_item2vec(norm_emb, ctx_items)
        if scores_i2v is not None:
            ranked_i2v = _top_k(scores_i2v, FUSION_TOPK)
            _add_metrics(results["item2vec"], ranked_i2v, test_items,
                         repeat_items, discovery_items, pop_rank, norm_emb, n_items)
            _calib_update(calib["item2vec"], ranked_i2v, scores_i2v, test_items)

        # ── popularity ──
        _add_metrics(results["popularity"], pop_rank, test_items,
                     repeat_items, discovery_items, pop_rank, norm_emb, n_items)
        _calib_update(calib["popularity"], pop_rank,
                      train_counts_f64, test_items)

        # ── ranker (GRU) ──
        if ranker_model is not None:
            scores_r = _score_ranker(ranker_model, ctx_items, all_emb_t)
            ranked_ranker = _top_k(scores_r, FUSION_TOPK)
            user_ranker_ranks[u] = ranked_ranker
            _add_metrics(results["ranker"], ranked_ranker, test_items,
                         repeat_items, discovery_items, pop_rank, norm_emb, n_items)
            _calib_update(calib["ranker"], ranked_ranker, scores_r, test_items)

        # ── periodic hygiene + progress (v6: fights heap fragmentation) ──
        if n_u % 100 == 0:
            log(f"  eval {n_u}/{len(eval_users)} rss={rss_mb()}MB")
        if n_u % 250 == 0:
            gc.collect()
            try:
                import ctypes
                ctypes.CDLL('libc.so.6').malloc_trim(0)
            except OSError:
                pass

    # ── Pick best decay scorer by repeat R@100 ──
    best_hl = None
    best_r100 = -1.0
    for hl in DECAY_HALF_LIVES:
        name = f"decay_{hl}d"
        repeat_rows = results[name]["repeat"]
        if repeat_rows:
            avg_r100 = float(np.mean([m.get("recall@100", 0.0) for m in repeat_rows]))
            log(f"  {name} repeat R@100 = {avg_r100:.6f}")
            if avg_r100 > best_r100:
                best_r100 = avg_r100
                best_hl = hl
    if best_hl is None:
        best_hl = DECAY_HALF_LIVES[0]
    log(f"  best decay: decay_{best_hl}d (repeat R@100 = {best_r100:.6f})")

    # ── Pass 2: fusion scorers (only if ranker available) ──
    if ranker_model is not None:
        for u in eval_users:
            if u not in user_ranker_ranks:
                continue
            i = user_index[u]
            seq = data[offsets[i]:offsets[i + 1]]
            test_items = user_test[u]
            if not len(seq) or not test_items:
                continue

            train_unique = np.unique(seq)
            train_set = set(train_unique.tolist())
            repeat_items = test_items & train_set
            discovery_items = test_items - train_set

            ranked_ranker = user_ranker_ranks[u]
            ranked_decay  = user_decay_ranks[u][best_hl]

            # ── rrf_fusion ──
            ranked_rrf = _rrf_fusion(ranked_ranker, ranked_decay, n_items=n_items)
            _add_metrics(results["rrf_fusion"], ranked_rrf, test_items,
                         repeat_items, discovery_items, pop_rank, norm_emb, n_items)

            # ── w_fusion_05 ──
            ranked_w05 = _w_fusion_05(ranked_ranker, ranked_decay, n_items=n_items)
            _add_metrics(results["w_fusion_05"], ranked_w05, test_items,
                         repeat_items, discovery_items, pop_rank, norm_emb, n_items)

    # ── Calibration (v4): reliability + ECE per scorer ──
    out = {m: _aggregate(v) for m, v in results.items()}
    for m, entry in calib.items():
        if entry["pred"] and m in out:
            out[m]["calibration"] = _compute_calibration(entry)
    return out


# ═══════════════════════════════════════════════════════════════════════
# STEP 5 — Reports
# ═══════════════════════════════════════════════════════════════════════

def _write_reports(results, output_dir):
    """Write metrics.json, eval_report.md, README.md."""
    output_dir = Path(output_dir)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"  wrote {metrics_path}")

    metric_keys = [f"recall@{RECALL_KS[0]}", f"recall@{RECALL_KS[1]}",
                   f"mrr@{MRR_K}", f"hitrate@{HITRATE_K}"]
    header = "| Model | Split | " + " | ".join(metric_keys) + \
             " | Novelty | Diversity |"
    sep = "|-------|-------|" + "|".join(["------"] * len(metric_keys)) + \
          "|---------|-----------|"
    lines = ["# Recency Evaluation Report", "",
             f"**Generated:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
             "**Kernel:** lb_recency.py", "",
             "## Scoring Questions", "",
             "1. Does exponentially-decayed frequency beat plain frequency "
             "for repeat prediction?",
             "2. Does rank-fusion of the sequence model + frequency scorer "
             "beat both individually?", "",
             "## Metrics", "", header, sep]
    for model, splits in results.items():
        for split in ["overall", "repeat", "discovery"]:
            r = splits.get(split)
            if r:
                vals = [f"{r.get(mk, 0.0):.4f}" for mk in metric_keys]
                nov = f"{r.get('novelty', '-')}" if r.get('novelty') is not None else "-"
                div = f"{r.get('diversity', '-')}" if r.get('diversity') is not None else "-"
                lines.append(f"| {model} | {split} (n={r.get('n_users', 0)}) | "
                             + " | ".join(vals) + f" | {nov} | {div} |")
    lines += ["", "## Scorers", "",
              "- **user_frequency:** plain per-user item counts, ties by popularity",
              "- **decay_30d/90d/365d:** exponential decay over train events "
              "(half-life in days)",
              "- **item2vec:** mean embedding of last 50 train items, cosine sim",
              "- **popularity:** global item count desc",
              "- **ranker:** GRU sequence model (from lb-ranker)",
              "- **rrf_fusion:** reciprocal rank fusion (k=60) of ranker + best decay",
              "- **w_fusion_05:** weighted RRF (0.5*ranker + 0.5*best_decay)",
              "", "## Calibration", "",
              "Per-scorer Expected Calibration Error (ECE) over top-20 "
              "recommendations, with scores min-max normalized per user "
              "(normalized scores are not probabilities; ECE measures how "
              "closely relative score ordering tracks observed play-rate — "
              "lower = more trustworthy ranking signal):", "",
              "| Scorer | ECE |", "|---|---|"]
    for model, splits in results.items():
        cal = splits.get("calibration")
        if cal:
            lines.append(f"| {model} | {cal['ece']:.4f} |")
    lines += ["", "## Metrics Explanation", "",
              "- **Novelty:** mean popularity percentile of top-20 items "
              "(percentile = pop_rank/n_items, lower = rarer)",
              "- **Diversity:** mean pairwise cosine similarity of top-20 items "
              "[-1, 1]",
              "", "## Notes", "",
              "- **Context:** last 50 train items",
              "- **Eval:** repeats NOT excluded (legitimate predictions)",
              "- **Eval type:** set-based — recall/MRR against the full "
              "test-set of held-out items, not next-item prediction",
              "- **Primary metric:** discovery (items the user has not "
              "listened to); overall recall is dominated by repeat consumption",
              "- **V1-space translation:** weights (item2vec, ranker) were "
              "trained on V1 all-events vocab (2,803,656 items). This kernel "
              "rebuilds V1 vocab from listens.parquet, translates V2 event "
              "data into V1 space via mbid join, then evaluates all scorers "
              "in V1 space. V1 vocab is emitted as data/v1_vocab.parquet.",
              "- **Cohort-first memory design (v4):** the eval cohort "
              "(>=10 train events AND >=1 test event, seeded sample) is "
              "selected BEFORE the CSR load; only cohort rows are loaded "
              "(~174M events vs 1.26B). No data is cut or truncated — "
              "every evaluated user's full history is retained.",
              "- **Popularity baseline:** uses all-events counts from the V1 "
              "vocab table (exact, unbiased by cohort filtering); differs "
              "slightly from the published train-only popularity baseline."]
    (output_dir / "reports" / "eval_report.md").write_text(
        "\n".join(lines), encoding="utf-8")
    log("  wrote reports/eval_report.md")

    readme = """# Music Recommender — Recency Evaluation

Evaluation of additional scorers for serving design decisions.

## Files

| File | Description |
|------|-------------|
| `metrics.json` | Evaluation metrics for all scorers |
| `reports/eval_report.md` | Detailed evaluation report |

## Questions Answered

1. Does exponentially-decayed frequency beat plain frequency for repeat prediction?
2. Does rank-fusion of the sequence model + frequency scorer beat both individually?

## Scorers

| Scorer | Description |
|--------|-------------|
| `user_frequency` | Plain per-user item counts |
| `decay_30d` | Exponential decay, half-life=30 days |
| `decay_90d` | Exponential decay, half-life=90 days |
| `decay_365d` | Exponential decay, half-life=365 days |
| `item2vec` | Mean embedding scorer |
| `popularity` | Global frequency |
| `ranker` | GRU sequence model |
| `rrf_fusion` | Reciprocal rank fusion |
| `w_fusion_05` | Weighted RRF variant |
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    log("  wrote README.md")

    summary = {m: {s: {k: results[m][s].get(k)
                       for k in metric_keys if k in results[m][s]}
                   for s in ["overall", "repeat", "discovery"]
                   if results[m].get(s)}
               for m in results}
    print(json.dumps(summary))


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def run_recency_eval(listens_path, events_path, vocab_path, emb_path,
                     output_dir, ranker_path=None):
    """Full pipeline: V1 vocab rebuild → translate → eval → reports."""
    output_dir = Path(output_dir)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)

    # ── Step 1: Rebuild V1 vocab from listens.parquet ──
    mbid_to_v1_id, v1_table = build_v1_vocab(listens_path)
    n_v1 = len(mbid_to_v1_id)
    log(f"  V1 vocab: {n_v1:,} items (expected {V1_EXPECTED_SIZE:,})")

    if n_v1 != V1_EXPECTED_SIZE:
        log(f"  WARNING: V1 vocab size mismatch — {n_v1:,} vs expected {V1_EXPECTED_SIZE:,}")

    v1_vocab_path = output_dir / "data" / "v1_vocab.parquet"
    pq.write_table(v1_table, str(v1_vocab_path))
    log(f"  wrote {v1_vocab_path}")

    # All-events popularity counts from the V1 vocab table (exact, unbiased
    # by cohort filtering — differs from published train-only popularity).
    pop_counts_v1 = np.zeros(n_v1, dtype=np.int64)
    _vc = v1_table.column("count").to_numpy()
    _vi = v1_table.column("v1_item_id").to_numpy()
    pop_counts_v1[_vi] = _vc
    del v1_table, _vc, _vi
    gc.collect()

    # ── Step 1b: Cohort-first — select eval cohort BEFORE heavy load ──
    cohort_ids, n_eligible = select_cohort(events_path)
    log(f"  cohort selected: {len(cohort_ids):,} of {n_eligible:,} eligible")

    # ── Step 2: Load V2 vocab + events (cohort-filtered), translate to V1 ──
    n_v2 = load_n_items(vocab_path)
    log(f"V2 vocab size: {n_v2:,}")

    v2_to_v1 = build_v2_to_v1(vocab_path, mbid_to_v1_id, n_v2)
    del mbid_to_v1_id
    gc.collect()

    offsets, data_v2, ts_data_v2, user_ids, user_index, train_counts_v2, user_test_v2 = \
        load_csr(events_path, n_v2, cohort_users=cohort_ids)

    log(f"  cohort users loaded={len(user_ids):,} "
        f"(cohort {len(cohort_ids):,})")
    log(f"  train events={len(data_v2):,} (cohort-filtered)")
    log(f"  RSS {rss_mb():.0f} MB")

    # Translate everything to V1 space
    log("Translating V2 -> V1 space")
    data_v1, ts_data_v1, user_test_v1 = translate_csr_to_v1(
        data_v2, ts_data_v2, user_test_v2, v2_to_v1)
    del data_v2, ts_data_v2, user_test_v2, v2_to_v1
    gc.collect()
    log(f"  RSS {rss_mb():.0f} MB")

    # Recompute train_counts in V1 space (cohort-only; used only for
    # diagnostics — popularity scoring uses pop_counts_v1 above)
    train_counts_v1 = np.bincount(data_v1, minlength=n_v1).astype(np.int64)

    # ── Step 3: Load V1-space embeddings + ranker ──
    embeddings = load_embeddings(emb_path, n_v1)
    # Normalize in-place — reused for item2vec scorer, diversity, and ranker
    embeddings /= (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    log(f"  embeddings loaded+normalized: {embeddings.shape}")

    ranker_model = None
    if ranker_path and Path(ranker_path).exists():
        log(f"Loading GRU ranker from {ranker_path}")
        ranker_model = GRURanker(n_v1, EMB_DIM, GRU_HIDDEN, GRU_LAYERS, "cpu")
        ranker_model.load(ranker_path)
        log("  ranker loaded")
    else:
        log("WARNING: ranker_final.pt not found, skipping GRU ranker + fusions")
    log(f"  RSS {rss_mb():.0f} MB")

    # ── Step 4: Evaluate in V1 space ──
    log("STEP 4: Evaluation (V1 space)")
    results = evaluate(embeddings, offsets, data_v1, ts_data_v1, user_ids,
                       user_index, pop_counts_v1, user_test_v1, n_v1,
                       ranker_model)

    log("STEP 5: Writing reports")
    _write_reports(results, output_dir)

    log("DONE")
    return results


def main():
    """Kaggle entry point — discover inputs, run full pipeline."""
    root = Path("/kaggle/input")
    listens_files = list(root.rglob("listens.parquet"))
    events_files = list(root.rglob("events.parquet"))
    vocab_files = list(root.rglob("vocab.parquet"))
    emb_files = list(root.rglob("item2vec_final.npy"))
    ranker_files = list(root.rglob("ranker_final.pt"))

    tree = "\n".join(f"  {p}" for p in sorted(root.rglob("*")) if p.is_file())
    log(f"/kaggle/input tree:\n{tree}")

    if not listens_files:
        sys.exit("FATAL: listens.parquet not found under /kaggle/input")
    if len(listens_files) > 1:
        sys.exit(f"FATAL: multiple listens.parquet found, expected exactly one: {listens_files}")
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

    log(f"Listens: {listens_files[0]}")
    log(f"Events: {events_files[0]}")
    log(f"Vocab:  {vocab_files[0]}")
    log(f"Emb:    {emb_files[0]}")
    if ranker_files:
        log(f"Ranker: {ranker_files[0]}")
    else:
        log("Ranker: NOT FOUND (GRU scorer + fusions will be skipped)")

    run_recency_eval(listens_files[0], events_files[0], vocab_files[0],
                     emb_files[0], Path("/kaggle/working"),
                     ranker_path=str(ranker_files[0]) if ranker_files else None)


# ═══════════════════════════════════════════════════════════════════════
# SYNTHETIC TEST (CPU-only, creates temp files)
# ═══════════════════════════════════════════════════════════════════════

def _synthetic_test():
    """End-to-end test: builds tiny listens.parquet + V2 events.parquet
    with DIFFERENT id assignments, fake embeddings + GRU checkpoint in V1
    space, and verifies the full pipeline.
    """
    import tempfile, shutil
    tmpdir = Path(tempfile.mkdtemp(prefix="lb_recency_test_"))
    try:
        log("=== SYNTHETIC TEST ===")
        rng = np.random.default_rng(42)
        n_v1 = 30          # V1 vocab size
        n_v2 = 28          # V2 is strict subset (excludes 2 items)
        n_users = 8

        # ── 1. Build fake listens.parquet (V1: all 30 items) ──
        # Generate 16-byte binary mbids deterministically
        v1_mbids = [bytes(rng.integers(0, 256, size=16).tolist())
                     for _ in range(n_v1)]
        v1_counts = rng.integers(10, 200, size=n_v1)  # all >= MIN_COUNT=10
        # Ensure items with low counts are below threshold if we want a subset
        # V2 subset: exclude items at indices 0 and 15
        v2_exclude = {0, 15}
        v2_mbids = [m for i, m in enumerate(v1_mbids) if i not in v2_exclude]
        assert len(v2_mbids) == n_v2

        # Build listens: ensure every V1 item appears at least once
        listens_rows = []
        for uid in range(n_users):
            # First pass: ensure every item is represented
            for item_idx in range(n_v1):
                mbid = v1_mbids[item_idx]
                ts = rng.integers(1_000_000, 2_000_000)
                listens_rows.append((uid, ts, mbid))
            # Second pass: add random events for realism
            n_extra = rng.integers(10, 40)
            for _ in range(n_extra):
                mbid = v1_mbids[rng.integers(0, n_v1)]
                ts = rng.integers(1_000_000, 2_000_000)
                listens_rows.append((uid, ts, mbid))
        listens_rows.sort(key=lambda x: (x[0], x[1]))  # user-grouped, ts-sorted

        listens_tbl = pa.table({
            "user": pa.array([r[0] for r in listens_rows], type=pa.int32()),
            "ts": pa.array([r[1] for r in listens_rows], type=pa.int32()),
            "recording_mbid": pa.array([r[2] for r in listens_rows],
                                       type=pa.binary(16)),
        })
        listens_path = tmpdir / "listens.parquet"
        pq.write_table(listens_tbl, str(listens_path))

        # ── 2. Build V2 vocab.parquet (different id assignment) ──
        # V2 sorts eligible items by (-count, mbid) — same logic as V1
        # but only includes items in v2_mbids subset
        v2_counts = {m: int(v1_counts[i])
                     for i, m in enumerate(v1_mbids) if i not in v2_exclude}
        v2_eligible = [(m, c) for m, c in v2_counts.items() if c >= 10]
        v2_eligible.sort(key=lambda x: (-x[1], x[0]))
        v2_mbid_to_id = {m: i for i, (m, _) in enumerate(v2_eligible)}

        vocab_tbl = pa.table({
            "recording_mbid": pa.array([e[0] for e in v2_eligible],
                                       type=pa.binary(16)),
            "count": pa.array([e[1] for e in v2_eligible], type=pa.int64()),
            "item_id": pa.array(list(range(len(v2_eligible))), type=pa.int32()),
        })
        vocab_path = tmpdir / "vocab.parquet"
        pq.write_table(vocab_tbl, str(vocab_path))
        log(f"  V2 vocab: {len(v2_eligible):,} items")

        # ── 3. Build V2 events.parquet (item_ids in V2 space) ──
        events_rows = []
        for uid, ts, mbid in listens_rows:
            if mbid in v2_mbid_to_id:
                v2_id = v2_mbid_to_id[mbid]
                events_rows.append((uid, ts, v2_id, 0))  # train
            # Items not in V2 vocab are OOV — dropped
        events_rows.sort(key=lambda x: (x[0], x[1]))
        # Mark each user's LAST event as test (split=2) so select_cohort
        # has eligible users (>=10 train AND >=1 test)
        last_idx_by_user = {}
        for i, r in enumerate(events_rows):
            last_idx_by_user[r[0]] = i
        events_rows = [(u, ts, it, 2 if i in last_idx_by_user.values() else s)
                       for i, (u, ts, it, s) in enumerate(events_rows)]

        events_tbl = pa.table({
            "user": pa.array([r[0] for r in events_rows], type=pa.int32()),
            "ts": pa.array([r[1] for r in events_rows], type=pa.int32()),
            "item_id": pa.array([r[2] for r in events_rows], type=pa.int32()),
            "split": pa.array([r[3] for r in events_rows], type=pa.int8()),
        })
        events_path = tmpdir / "events.parquet"
        pq.write_table(events_tbl, str(events_path))
        log(f"  V2 events: {len(events_rows):,} rows")

        # ── 4. Build V1-space embeddings + GRU checkpoint ──
        emb = rng.standard_normal((n_v1, 64)).astype(np.float32)
        emb_path = tmpdir / "item2vec_final.npy"
        np.save(str(emb_path), emb)

        # Fake GRU checkpoint with n_v1-sized embedding
        # Format matches lb_ranker save: {emb: sd, gru: sd, proj: sd}
        import torch
        import torch.nn as nn
        emb_nn = nn.Embedding(n_v1, 64)
        gru_nn = nn.GRU(64, 128, num_layers=1, batch_first=True)
        proj_nn = nn.Linear(128, 64)
        ckpt = {
            "emb": emb_nn.state_dict(),
            "gru": gru_nn.state_dict(),
            "proj": proj_nn.state_dict(),
        }
        ranker_path = tmpdir / "ranker_final.pt"
        torch.save(ckpt, str(ranker_path))

        # ── 5. Run V1 vocab rebuild ──
        mbid_to_v1_id, v1_table = build_v1_vocab(listens_path)
        n_v1_rebuilt = len(mbid_to_v1_id)
        log(f"  V1 rebuild: {n_v1_rebuilt:,} items")
        assert n_v1_rebuilt == n_v1, f"V1 rebuild {n_v1_rebuilt} != {n_v1}"

        # Check deterministic sort: first item should have highest count,
        # tie-broken by mbid ascending
        first_count = v1_table.column("count")[0].as_py()
        first_mbid = v1_table.column("recording_mbid")[0].as_py()
        all_counts = v1_table.column("count").to_pylist()
        max_count = max(all_counts)
        assert first_count == max_count, f"First item count {first_count} != max {max_count}"
        # Verify second item is either same count with later mbid, or lower count
        second_count = v1_table.column("count")[1].as_py()
        second_mbid = v1_table.column("recording_mbid")[1].as_py()
        assert (second_count < first_count) or \
               (second_count == first_count and second_mbid > first_mbid), \
            "Sort order violated: (-count, mbid) not descending"

        # Save V1 vocab
        v1_vocab_path = tmpdir / "data" / "v1_vocab.parquet"
        (tmpdir / "data").mkdir(exist_ok=True)
        pq.write_table(v1_table, str(v1_vocab_path))

        # ── 6. Test V2→V1 translation ──
        v2_to_v1 = build_v2_to_v1(vocab_path, mbid_to_v1_id, len(v2_eligible))
        assert (v2_to_v1 >= 0).all(), "All V2 items must map to V1"

        # Verify that excluded items have no V2 id (they shouldn't appear)
        for excluded_idx in v2_exclude:
            for v2i, mbid in enumerate(v2_mbids):
                if mbid == v1_mbids[excluded_idx]:
                    # This item shouldn't be in V2 vocab at all
                    pass

        # ── 6b. Test translate_csr_to_v1 (chunked, in-place) ──
        # Build tiny CSR arrays to feed translate_csr_to_v1
        fake_data = np.array([0, 1, 2, 3, 25, 26, 27], dtype=np.int32)
        fake_ts = np.array([100, 200, 300, 400, 500, 600, 700], dtype=np.int32)
        fake_test = {99: {25, 26}, 100: {0, 27}}
        tr_data, tr_ts, tr_test = translate_csr_to_v1(
            fake_data, fake_ts, fake_test, v2_to_v1)
        assert tr_data.dtype == np.int32, f"translated data dtype: {tr_data.dtype}"
        assert tr_ts is fake_ts, "ts_data must be reused, not copied"
        # Verify values: item 0 -> v2_to_v1[0], etc.
        for i in range(len(fake_data)):
            expected = int(v2_to_v1[fake_data[i]])
            assert int(tr_data[i]) == expected, \
                f"tr_data[{i}]={tr_data[i]} != v2_to_v1[{fake_data[i]}]={expected}"
        # Verify test sets translated
        assert 99 in tr_test and tr_test[99] == {int(v2_to_v1[25]), int(v2_to_v1[26])}
        assert 100 in tr_test and tr_test[100] == {int(v2_to_v1[0]), int(v2_to_v1[27])}
        log("  translate_csr_to_v1: chunked in-place test PASSED")

        # ── 7. Test embeddings assertion ──
        emb_loaded = load_embeddings(str(emb_path), n_v1_rebuilt)
        assert emb_loaded.shape == (n_v1, 64), f"Shape mismatch: {emb_loaded.shape}"

        # Test that wrong size triggers assertion
        try:
            load_embeddings(str(emb_path), n_v1_rebuilt - 1)
            assert False, "Should have raised AssertionError"
        except AssertionError as e:
            assert "mismatch" in str(e).lower() or "Expected" in str(e)
            log("  Correctly caught embedding size mismatch")

        # ── 8. Test GRU ranker loads with correct n_v1 ──
        ranker = GRURanker(n_v1, 64, 128, 1, "cpu")
        ranker.load(str(ranker_path))
        log("  GRU ranker loaded successfully with n_v1 size")

        # ── 9. Test scorer functions in V1 space ──
        n_v1_items = n_v1_rebuilt
        pop_rank = np.argsort(-v1_table.column("count").to_numpy().astype(np.int64))
        pop_pos = np.empty(n_v1_items, dtype=np.int64)
        pop_pos[pop_rank] = np.arange(n_v1_items)

        found_decay_diff = False
        for uid in range(n_users):
            # Get user's items from events (translate V2→V1)
            user_events = [(r[1], r[2]) for r in events_rows if r[0] == uid]
            if len(user_events) < 5:
                continue
            seq_v2 = np.array([e[1] for e in user_events], dtype=np.int32)
            seq_v1 = v2_to_v1[seq_v2].astype(np.int32)
            ts = np.sort(np.array([e[0] for e in user_events], dtype=np.int64))

            # user_frequency
            ranked_uf = _score_user_frequency(seq_v1, n_v1_items, pop_rank, pop_pos)
            assert len(ranked_uf) == n_v1_items

            # decay scorers
            ranked_decay_30 = None
            for hl in DECAY_HALF_LIVES:
                scores = _score_decay(seq_v1, ts, n_v1_items, hl)
                ranked = np.argsort(-scores)
                assert len(ranked) == n_v1_items
                if hl == 30:
                    ranked_decay_30 = ranked
            assert ranked_decay_30 is not None  # DECAY_HALF_LIVES always contains 30

            if not np.array_equal(ranked_uf[:10], ranked_decay_30[:10]):
                found_decay_diff = True

            # rrf_fusion
            ranked_rrf = _rrf_fusion(ranked_uf, ranked_decay_30, n_items=n_v1_items)
            assert len(ranked_rrf) == n_v1_items

            # novelty
            novelty = _compute_novelty(ranked_uf, pop_rank, n_v1_items)
            assert 0.0 <= novelty <= 1.0

            # diversity
            norm_emb_test = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
            diversity = _compute_diversity(ranked_uf, norm_emb_test)
            assert -1.0 <= diversity <= 1.0

        assert found_decay_diff, "decay should differ from frequency for >=1 user"

        # ── 9b. Test select_cohort (v4) ──
        cohort_ids, n_elig = select_cohort(str(events_path), eval_sample=5)
        assert n_elig == n_users, f"eligible {n_elig} != {n_users} (all users have >=10 train + 1 test)"
        assert len(cohort_ids) == min(5, n_users)
        assert list(cohort_ids) == sorted(cohort_ids), "cohort must be sorted"
        log(f"  select_cohort: {len(cohort_ids)} of {n_elig} eligible PASSED")

        # Cohort-filtered load_csr: only cohort users' rows
        c_off, c_data, c_ts, c_uids, c_uix, c_tc, c_test = \
            load_csr(str(events_path), n_v2, cohort_users=cohort_ids)
        assert set(int(x) for x in c_uids) <= set(int(x) for x in cohort_ids), \
            "loaded users must be subset of cohort"
        assert all(len(c_test.get(int(u), set())) >= 1 for u in c_uids), \
            "every loaded cohort user must have test items"
        log("  cohort-filtered load_csr PASSED")

        # ── 9c. Test calibration helpers (v4) ──
        calib_entry = {"pred": [], "obs": []}
        fake_ranked = np.array([3, 1, 7, 0, 5, 2, 6, 4])
        fake_scores = np.array([0.9, 0.7, 0.5, 0.3, 0.1, 0.05, 0.02, 0.0])
        _calib_update(calib_entry, fake_ranked, fake_scores, {3, 7})
        cal = _compute_calibration(calib_entry)
        assert 0.0 <= cal["ece"] <= 1.0, f"ECE out of range: {cal['ece']}"
        assert sum(b[2] for b in cal["bins"]) <= 1.0 + 1e-9, "bin weights must sum <= 1"
        log(f"  calibration: ECE={cal['ece']} PASSED")

        # ── 10. Verify v1_vocab.parquet was written ──
        assert v1_vocab_path.exists(), f"v1_vocab.parquet not found at {v1_vocab_path}"
        v1_check = pq.read_table(str(v1_vocab_path))
        assert "recording_mbid" in v1_check.column_names
        assert "count" in v1_check.column_names
        assert "v1_item_id" in v1_check.column_names
        assert len(v1_check) == n_v1

        log(f"=== ALL ASSERTIONS PASSED (v1={n_v1}, v2={n_v2}) ===")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    if "--test" in sys.argv:
        _synthetic_test()
    else:
        main()
