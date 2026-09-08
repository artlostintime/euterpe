"""
lb-eval v5.0.0 — unified evaluation kernel (22 scorers) + train-only
candidate universe + split-labeled bootstrap CIs + saturation curve.

Evaluates whether exponentially-decayed frequency beats plain frequency for
repeat prediction, and whether rank-fusion of the sequence model + frequency
scorer beats both individually. Also scores a BPR-MF implicit-feedback
baseline (from lb-bpr) under the IDENTICAL protocol — the published lb-bpr
metrics came from its own internal eval (whose @20/@100 metrics share one
top-100 set, and whose cohort/protocol differ), so all cross-scorer
comparisons in this paper must come from THIS kernel only.

VERSION HISTORY:
  v1.0.0 — 8 scorers, BPR added under unified protocol (kernel v1)
  v2.0.0 — + NDCG@K / Precision@K (kernel v2)
  v3.0.0 — master-plan analyses: per-user metrics parquet + percentiles,
           activity segments, long-tail head/mid/tail, paired bootstrap
           CIs, popularity_train leakage control, decay functional forms,
           fusion weight sweep (replaces w_fusion_05 — mathematically
           identical to rrf_fusion), catalog coverage + repeat share
           (kernel v3)
  v4.0.0 — review-response protocol (kernel v4):
             * validation-based selection: decay half-life + fusion weight
               selected on VAL split (train-only history), never on TEST
             * test evaluation uses train+val history (fixes blind-gap
               recency handicap)
             * user_frequency ties by item ID (deterministic, no future info)
             * popularity_train is the PRIMARY popularity baseline;
               all-events popularity = leakage-sensitivity variant
             * discovery decomposed into globally-known vs cold items
             * item2vec profile ablations: unique-mean, recency-weighted,
               windows 10/100
             * item-item kNN baseline (co-occurrence graph, top-20
               neighbors/item)
             * per-user history-saturation diagnostic
            v2/v3 headline numbers are NOT comparable to v4 (protocol
            changed); the paper reports v4.
  v4.1.0 — robustness round (kernel v5):
             * multi-seed: cohort re-selected under seeds 42/43/44;
               per-scorer mean +/- sd across seeds reported
             * global-chronological-split sensitivity: test = events
               after the cohort's global 14-day cutoff (vs per-user
               windows); reported as sensitivity, not a second
               benchmark
  v5.0.0 — review-2 response (kernel v6):
             * TRAIN-ONLY CANDIDATE UNIVERSE (primary): the candidate
               set is the V2 trainprep vocab (train-split counts >= 10,
               2,787,934 items) — items can no longer enter the catalog
               via future (test-window) interactions. The previous
               all-events V1 universe (2,803,656) is retained as a
               FIXED-CATALOG SENSITIVITY variant (v4/v5 numbers become
               the sensitivity numbers).
             * popularity_train_global: train-split counts over ALL
               ~37K users (not cohort-only) — the new PRIMARY popularity
               baseline; cohort-only popularity_train and all-events
               popularity become leakage diagnostics.
             * split-labeled bootstrap CIs: the 5 key pairs are
               bootstrapped on overall AND repeat AND discovery
               recall@100 per-user vectors, with the split named in
               each key (fixes the §6.8 split-mismatch inconsistency).
             * saturation curve: per-user proportion of top-K items
               already in history, K in {10,20,50,100,200,500}, for the
               VAL-selected decay scorer (empirical saturation
               mechanism figure).
             * per-user metric parquet extended: all primary scorers'
               recall@100 per split (feeds §6.8 regeneration).

VERSION-SKEW NOTE: item2vec_final.npy and ranker_final.pt were trained on
trainprep V1 vocab (ALL-events counts, 2,803,656 items). The current V2
trainprep output (events.parquet + vocab.parquet) uses a TRAIN-ONLY vocab
(~2,787,934 items) with different item_id assignments. This kernel rebuilds
the V1 vocab from listens.parquet (sort: -count, then mbid ascending for
deterministic id assignment), translates V2 event data into V1 space via
mbid-keyed join, then runs all scorers in V1 space. V1 vocab is emitted
as data/v1_vocab.parquet for pairing with the published weights.
In v5 the V2 train-only vocab ALSO defines the candidate universe mask
(train_universe_mask over V1 ids); scorers rank within the universe, and
test items outside it are excluded from metrics (universe coverage is
reported).

COHORT-FIRST (since v1): the eval cohort is selected FIRST (same
eligibility rule and seed) and only cohort users' events are materialized
— full precision, no data cut for any evaluated user; peak RSS ~4 GB.

CALIBRATION: per-scorer reliability curves + expected calibration error
(ECE) over min-max-normalized top-20 scores — measures ranking-score
trustworthiness, not the probability that any single recommendation is
"right".

Scorers (22):
  - user_frequency: plain per-user item counts (ties by item ID)
  - decay_30d / decay_90d / decay_365d: exponential decay half-lives
  - decay_power1 / decay_linear365 / decay_step_{7,30,90,365}d: functional
    forms (robustness)
  - popularity_train_global: train-split counts over ALL users (PRIMARY
    popularity baseline, v5)
  - popularity_train: cohort-train-only counts (leakage diagnostic)
  - popularity: all-events counts (leakage-sensitivity variant)
  - item2vec: mean last-50 embedding cosine
  - item2vec_unique / item2vec_recency / item2vec_w10 / item2vec_w100:
    profile ablations (robustness)
  - item_item_knn: co-occurrence kNN baseline (top-20 neighbors/item)
  - ranker: GRU sequence model (loaded from pre-trained checkpoint)
  - bpr_mf: BPR-MF factors (from lb-bpr)
  - rrf_fusion: weighted RRF (ranker + VAL-selected decay, VAL-selected w)
  - fusion_sweep: w in 0..1 over the same pair (analysis.json)

Inputs (rglob under /kaggle/input):
  listens.parquet  (user, ts, recording_mbid, ... ) — from lb-sanitize
  events.parquet   (user, ts, item_id, split)      — V2 trainprep output
  vocab.parquet    (recording_mbid, count, item_id) — V2 vocab
  item2vec_final.npy  ((2803656, 64) f32) — V1-space embeddings
  ranker_final.pt     (GRURanker checkpoint, optional) — V1-space
  bpr_final.pt        (BPR-MF state_dict, optional) — V2-space; user_factors
                     rows follow sorted ALL-train-user ids (same set this
                     kernel sees in select_cohort), item_factors rows are
                     V2 item_ids (translated to V1 via vocab mbid join)

Outputs: data/v1_vocab.parquet, metrics.json, analysis.json,
         data/per_user_metrics.parquet, reports/eval_report.md, README.md

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
MULTI_SEEDS     = [42, 43, 44]     # v4.1: cohort seeds for robustness round
RRF_K           = 60            # RRF constant
DECAY_HALF_LIVES = [30, 90, 365]  # half-lives in days
STEP_WINDOWS    = [7, 30, 90, 365]  # step-decay window sizes in days (#6)
BOOTSTRAP_B     = 1000          # paired bootstrap resamples (#25)
FUSION_SWEEP_WS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # (#16)
TAIL_HEAD_PCT  = 0.01          # head = top 1% by all-events popularity (#11)
TAIL_MID_PCT   = 0.10          # mid = top 1-10%; tail = bottom 90%
SATURATION_KS  = [10, 20, 50, 100, 200, 500]  # v5: saturation curve Ks

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
    """Load train+val splits into CSR arrays; collect test item sets in pass 1.

    If cohort_users is given (array of user ids), only those users' rows are
    loaded — the eval cohort is selected BEFORE this call (cohort-first
    memory design: ~5K users ≈ 174M events ≈ 1.4GB vs 10GB full load).

    v4: split==1 (validation) rows are loaded into the SAME CSR, in file
    (= ts) order, with a per-user boundary array `val_start` marking where
    validation begins. Train history = [offsets[i], val_start[i]);
    train+val history = [offsets[i], offsets[i+1]). This enables
    validation-based selection and train+val test evaluation.

    Returns:
      offsets:     int64 (n_users+1,) — user's train+val events are
                   data[offsets[i]:offsets[i+1]], file order (= ts order)
      data:        int32 concatenated train+val item_ids
      ts_data:     int32 concatenated train+val timestamps (parallel to data)
      user_ids:    int32 sorted unique user ids
      user_index:  dict user_id -> CSR row index
      train_counts:int64 (n_items,) per-item train-ONLY frequency (padded)
      user_test:   dict user_id -> set(test item_ids)
      user_val:    dict user_id -> set(val item_ids)
      val_start:   int64 (n_users,) — CSR index where user's val rows begin
    """
    cohort_arr = (np.asarray(sorted(cohort_users), dtype=np.int32)
                  if cohort_users is not None else None)
    log(f"Loading events from {events_path}"
        + (f" (cohort-filtered: {len(cohort_arr):,} users)" if cohort_arr is not None else ""))
    pf = pq.ParquetFile(str(events_path))

    # ── Pass 1: per-user train/val counts + test/val sets + order check ──
    user_train_counts = defaultdict(int)   # 37K entries
    user_val_counts = defaultdict(int)     # v4
    user_test = defaultdict(set)            # ~5M entries total
    user_val = defaultdict(set)            # v4
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

        m = splits_np == 1                        # v4: validation rows
        if m.any():
            u, c = np.unique(users_np[m], return_counts=True)
            for uid, cnt in zip(u.tolist(), c.tolist()):
                user_val_counts[uid] += cnt
            u = users_np[m]; it = items_np[m]
            for uid in np.unique(u).tolist():
                user_val[uid].update(it[u == uid].tolist())

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

    # ── Build CSR skeleton (train+val rows) ──
    user_ids = np.array(sorted(user_train_counts.keys()), dtype=np.int32)
    user_index = {int(u): i for i, u in enumerate(user_ids.tolist())}
    offsets = np.zeros(len(user_ids) + 1, dtype=np.int64)
    for i, u in enumerate(user_ids.tolist()):
        offsets[i + 1] = offsets[i] + user_train_counts[u] + user_val_counts.get(u, 0)
    data = np.empty(int(offsets[-1]), dtype=np.int32)
    ts_data = np.empty(int(offsets[-1]), dtype=np.int32)
    # v4: per-user boundary — train rows occupy [offsets[i], offsets[i]+n_train),
    # val rows follow within the same user block.
    val_start = np.zeros(len(user_ids), dtype=np.int64)
    for i, u in enumerate(user_ids.tolist()):
        val_start[i] = offsets[i] + user_train_counts[u]

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
        m = (splits_np == 0) | (splits_np == 1)   # v4: train + val rows
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

    # train-ONLY counts (exclude val rows) for popularity_train + eligibility
    train_only = np.zeros(len(data), dtype=bool)
    for i in range(len(user_ids)):            # vectorized per-user slice marking
        train_only[val_start[i]:offsets[i + 1]] = False
        train_only[offsets[i]:val_start[i]] = True
    train_counts = np.bincount(data[train_only], minlength=n_items).astype(np.int64)
    log(f"  CSR: {len(data):,} train+val events ({int(train_only.sum()):,} train / "
        f"{len(data) - int(train_only.sum()):,} val), {len(user_ids):,} users, "
        f"max_item_id={int(data.max()) if len(data) else -1}")
    return (offsets, data, ts_data, user_ids, user_index, train_counts,
            user_test, user_val, val_start)


def load_n_items(vocab_path):
    """Vocab size from vocab.parquet (item_ids are dense 0..N-1 by construction)."""
    vt = pq.read_table(str(vocab_path), columns=["item_id"])
    ids = vt.column("item_id").to_numpy()
    return int(ids.max()) + 1


def select_cohort(events_path, eval_sample=EVAL_SAMPLE, seed=SEED):
    """Cohort-first design: pick the eval cohort BEFORE the heavy CSR load.

    Cheap pass over (user, item_id, split) — eligibility mirrors evaluate():
    >=10 train events AND >=1 test event. Returns (cohort_ids int32 sorted,
    n_eligible, all_train_users int32 sorted). all_train_users is every user
    with >=1 train event — exactly the user set lb-bpr's CSR (and therefore
    its user_factors row order) was built over.
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
    all_train_users = np.array(sorted(train_counts.keys()), dtype=np.int32)
    log(f"  eligible users: {len(eligible):,} -> cohort {len(cohort):,} "
        f"(all train users: {len(all_train_users):,})")
    return (np.array(sorted(cohort), dtype=np.int32), len(eligible),
            all_train_users)


def load_embeddings(npy_path, n_items):
    """Load frozen item2vec embeddings. Returns (n_items, 64) float32 numpy."""
    emb = np.load(str(npy_path)).astype(np.float32)
    assert emb.shape[1] == EMB_DIM, f"Expected dim {EMB_DIM}, got {emb.shape}"
    assert emb.shape[0] == n_items, \
        (f"Embedding vocab mismatch: npy has {emb.shape[0]:,} items, "
         f"expected {n_items:,}. Check that item2vec was trained on V1 vocab.")
    return emb


def load_bpr(bpr_path, v2_to_v1, all_train_users, n_v1):
    """Load BPR-MF factors from lb-bpr; translate item factors V2 -> V1.

    lb-bpr's CSR (and therefore user_factors row order) was built over ALL
    users with >=1 train event, sorted by user id — asserted against
    all_train_users from select_cohort (same events.parquet, same rule).
    item_factors rows are V2 item_ids; translated to V1 via v2_to_v1
    (V2 ⊂ V1 by mbid, so unmapped rows should not exist; any that do get
    zero vectors and a warning).

    Returns (user_factors (n_all_users, 64) f32 aligned to all_train_users,
    item_factors_v1 (n_v1, 64) f32).
    """
    import torch
    sd = torch.load(str(bpr_path), map_location="cpu", weights_only=True)
    uf = sd["user_factors.weight"].numpy().astype(np.float32)
    itf_v2 = sd["item_factors.weight"].numpy().astype(np.float32)
    assert uf.shape[0] == len(all_train_users), (
        f"BPR user_factors rows {uf.shape[0]:,} != all-train-users "
        f"{len(all_train_users):,} — lb-bpr CSR user set mismatch")
    assert itf_v2.shape[0] == len(v2_to_v1), (
        f"BPR item_factors rows {itf_v2.shape[0]:,} != V2 vocab size "
        f"{len(v2_to_v1):,}")
    item_factors_v1 = np.zeros((n_v1, itf_v2.shape[1]), dtype=np.float32)
    mapped = v2_to_v1 >= 0
    item_factors_v1[v2_to_v1[mapped]] = itf_v2[mapped]
    n_unmapped = int((~mapped).sum())
    if n_unmapped:
        log(f"  WARNING: {n_unmapped:,} V2 items unmapped to V1 "
            f"(zero factor vectors)")
    log(f"  BPR factors: {uf.shape[0]:,} users, "
        f"{int(mapped.sum()):,}/{len(v2_to_v1):,} V2 items translated to V1")
    return uf, item_factors_v1


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


def _score_item2vec_unique(norm_emb, ctx_items):
    """Ablation: mean over UNIQUE context items (dedupe repeats)."""
    uniq = np.unique(ctx_items)
    ctx_vec = norm_emb[uniq].mean(axis=0)
    nrm = np.linalg.norm(ctx_vec)
    if nrm > 1e-8:
        ctx_vec = ctx_vec / nrm
        return norm_emb @ ctx_vec
    return None


def _score_item2vec_recency(norm_emb, ctx_items, ctx_ts, ref_ts,
                             half_life_days=90.0):
    """Ablation: recency-weighted mean of context embeddings (exp decay)."""
    age_days = np.maximum((ref_ts - ctx_ts) / 86400.0, 0.0)
    w = 0.5 ** (age_days / half_life_days)
    vecs = norm_emb[ctx_items] * w[:, None]
    ctx_vec = vecs.sum(axis=0) / (w.sum() + 1e-12)
    nrm = np.linalg.norm(ctx_vec)
    if nrm > 1e-8:
        ctx_vec = ctx_vec / nrm
        return norm_emb @ ctx_vec
    return None


def _score_item2vec_window(norm_emb, seq_items, window):
    """Ablation: mean over last-`window` events (raw sequence, no dedupe)."""
    ctx = seq_items[-window:]
    ctx_vec = norm_emb[ctx].mean(axis=0)
    nrm = np.linalg.norm(ctx_vec)
    if nrm > 1e-8:
        ctx_vec = ctx_vec / nrm
        return norm_emb @ ctx_vec
    return None


def _score_popularity(pop_rank):
    """Popularity scorer: global count desc (same for all users)."""
    return pop_rank


def _score_user_frequency(seq, n_items, pop_rank, pop_pos):
    """User-frequency scorer: user freq desc, ties by item ID asc (v4:
    deterministic, no future information in the tie-break)."""
    uf = np.bincount(seq, minlength=n_items)
    user_items = np.where(uf > 0)[0]
    order1 = user_items[np.lexsort((user_items, -uf[user_items]))]
    rest_mask = np.ones(n_items, dtype=bool)
    rest_mask[user_items] = False
    return np.concatenate([order1, pop_rank[rest_mask[pop_rank]]])


def _uf_scores(seq, n_items, pop_pos):
    """Raw user-frequency scores for calibration: freq + tiny ID tie-break."""
    uf = np.bincount(seq, minlength=n_items).astype(np.float64)
    return uf + 1e-9 * (n_items - np.arange(n_items))


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


def _score_decay_power(seq_items, seq_ts, n_items, exponent=1.0):
    """Power-law decay: w = 1/(1+age_days)^exponent (#6)."""
    max_ts = seq_ts.max()
    age_days = (max_ts - seq_ts).astype(np.float64) / 86400.0
    weights = 1.0 / (1.0 + age_days) ** exponent
    scores = np.zeros(n_items, dtype=np.float64)
    np.add.at(scores, seq_items, weights)
    return scores


def _score_decay_linear(seq_items, seq_ts, n_items, horizon_days=365.0):
    """Linear decay: w = max(0, 1 - age_days/horizon) (#6)."""
    max_ts = seq_ts.max()
    age_days = (max_ts - seq_ts).astype(np.float64) / 86400.0
    weights = np.clip(1.0 - age_days / horizon_days, 0.0, None)
    scores = np.zeros(n_items, dtype=np.float64)
    np.add.at(scores, seq_items, weights)
    return scores


def _score_decay_step(seq_items, seq_ts, n_items, window_days):
    """Step decay: w = 1 if age_days <= window else 0 (#6)."""
    max_ts = seq_ts.max()
    age_days = (max_ts - seq_ts).astype(np.float64) / 86400.0
    weights = (age_days <= window_days).astype(np.float64)
    scores = np.zeros(n_items, dtype=np.float64)
    np.add.at(scores, seq_items, weights)
    return scores


def _top_k(scores, k=200, mask=None):
    """Top-k item ids by score, descending. O(n) select + small sort — avoids
    full 2.8M argsort churn per scorer per user (metrics need ≤top-100,
    fusion needs top-200). k clamped to array size for tiny catalogs.

    v5: optional boolean mask restricts candidates to the train-only
    universe (universe-masked ranking)."""
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    k = min(k, scores.shape[0] - 1)
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


# ═══════════════════════════════════════════════════════════════════════
# STEP 3 — Metrics
# ═══════════════════════════════════════════════════════════════════════

def _compute_metrics(ranked, test_items):
    """Recall@K, Precision@K, NDCG@K, MRR@K, HitRate@K for one user."""
    test_set = set(test_items)
    result = {}
    for k in RECALL_KS:
        top_k = set(ranked[:k].tolist())
        hits = len(top_k & test_set)
        result[f"recall@{k}"] = (hits / len(test_set)
                                 if test_set else 0.0)
        result[f"precision@{k}"] = hits / k
        # NDCG@K: binary gains, ideal DCG = min(|test|, k) ones
        dcg = 0.0
        for rank, item in enumerate(ranked[:k]):
            if int(item) in test_set:
                dcg += 1.0 / np.log2(rank + 2)
        ideal_hits = min(len(test_set), k)
        idcg = sum(1.0 / np.log2(r + 2) for r in range(ideal_hits))
        result[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0
    rr = 0.0
    for rank, item in enumerate(ranked[:MRR_K]):
        if int(item) in test_set:
            rr = 1.0 / (rank + 1)
            break
    result[f"mrr@{MRR_K}"] = rr
    top_k = set(ranked[:HITRATE_K].tolist())
    result[f"hitrate@{HITRATE_K}"] = 1.0 if (top_k & test_set) else 0.0
    return result


def _compute_novelty(ranked, pop_pos, n_items, k=20):
    """Mean popularity percentile of top-k items. Lower = rarer/more novel.

    v5 fix: takes pop_pos (full-length array, pop_pos[item] = its popularity
    rank) — the old code indexed the rank LIST by item id, which only worked
    accidentally because V1 ids are count-sorted (id == rank) and broke when
    the universe mask compressed pop_rank."""
    top_k = ranked[:k]
    return float(np.mean(pop_pos[top_k].astype(np.float64) / n_items))


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
                 pop_pos, norm_emb, n_items, known_mask=None):
    """Compute all metrics for one user and append to buckets.

    v4: discovery is decomposed into globally-known vs cold items when
    `known_mask` (bool array over item ids) is provided (#4).
    v5: takes pop_pos (rank-position array) for novelty — see
    _compute_novelty.
    """
    novelty = _compute_novelty(ranked, pop_pos, n_items)
    diversity = _compute_diversity(ranked, norm_emb)

    o = _compute_metrics(ranked, test_items)
    o["novelty"] = novelty
    o["diversity"] = diversity
    bucket["overall"].append(o)

    if repeat_items:
        bucket["repeat"].append(_compute_metrics(ranked, repeat_items))
    if discovery_items:
        bucket["discovery"].append(_compute_metrics(ranked, discovery_items))
        if known_mask is not None:
            known = {it for it in discovery_items if known_mask[it]}
            cold = discovery_items - known
            if known:
                bucket["discovery_known"].append(
                    _compute_metrics(ranked, known))
            if cold:
                bucket["discovery_cold"].append(
                    _compute_metrics(ranked, cold))


def _aggregate(per_user_metrics):
    """Average per-user metrics across cohort."""
    agg = {}
    for split in ["overall", "repeat", "discovery",
                  "discovery_known", "discovery_cold"]:
        rows = per_user_metrics.get(split)
        if not rows:
            agg[split] = {}
            continue
        metric_keys = [f"recall@{k}" for k in RECALL_KS] + \
            [f"precision@{k}" for k in RECALL_KS] + \
            [f"ndcg@{k}" for k in RECALL_KS] + \
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


# ═══════════════════════════════════════════════════════════════════════
# STEP 4b — V3 analysis helpers (master plan)
# ═══════════════════════════════════════════════════════════════════════

def _percentiles(vals):
    """p10/25/50/75/90 of a list of floats (#2)."""
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return {}
    return {f"p{p}": round(float(np.percentile(a, p)), 6)
            for p in (10, 25, 50, 75, 90)}


def _activity_segment(n_train_events):
    """Coarse activity bucket (#9)."""
    if n_train_events < 200:
        return "light"
    if n_train_events < 1000:
        return "medium"
    return "heavy"


def _tail_bucket(item_id, pop_rank_pos, n_items):
    """head/mid/tail by all-events popularity rank (#11).

    pop_rank_pos: array mapping item_id -> popularity rank position
    (0 = most popular). head = top 1%, mid = next 9%, tail = rest.
    """
    pct = pop_rank_pos[item_id] / n_items
    if pct < TAIL_HEAD_PCT:
        return "head"
    if pct < TAIL_MID_PCT:
        return "mid"
    return "tail"


def _paired_bootstrap(metric_a, metric_b, b=BOOTSTRAP_B, seed=SEED):
    """Paired bootstrap CI for mean difference a-b (#25).

    Returns {"mean_diff", "ci_lo", "ci_hi", "p_gt0"} — p_gt0 is the
    fraction of resamples with positive difference (one-sided
    sign-flip probability, reported as a descriptive stat).
    """
    a = np.asarray(metric_a, dtype=np.float64)
    c = np.asarray(metric_b, dtype=np.float64)
    n = len(a)
    if n == 0 or n != len(c):
        return {}
    rng = np.random.default_rng(seed)
    diffs = a - c
    idx = rng.integers(0, n, size=(b, n))
    boot = diffs[idx].mean(axis=1)
    return {
        "mean_diff": round(float(diffs.mean()), 6),
        "ci_lo": round(float(np.percentile(boot, 2.5)), 6),
        "ci_hi": round(float(np.percentile(boot, 97.5)), 6),
        "p_gt0": round(float((boot > 0).mean()), 4),
    }


def _weighted_rrf(ranked_a, ranked_b, w, k=RRF_K, n_items=None):
    """Weighted RRF: w/(k+rank_a) + (1-w)/(k+rank_b). w=0.5 ≈ rrf_fusion."""
    if n_items is None:
        n_items = max(int(ranked_a.max()), int(ranked_b.max())) + 1
    ranks_a = np.full(n_items, float(len(ranked_a)), dtype=np.float64)
    ranks_b = np.full(n_items, float(len(ranked_b)), dtype=np.float64)
    ranks_a[ranked_a] = np.arange(len(ranked_a), dtype=np.float64)
    ranks_b[ranked_b] = np.arange(len(ranked_b), dtype=np.float64)
    scores = w / (k + ranks_a + 1) + (1.0 - w) / (k + ranks_b + 1)
    return np.argsort(-scores)


def _fused_top_k(ranked_a, ranked_b, w, k, topk=200,
                 rank_scratch_a=None, rank_scratch_b=None):
    """Top-K of the weighted-RRF fusion WITHOUT the full argsort.

    Only the union of the two top-`topk` lists can appear in the fused
    top-K (out-of-list scores are strictly lower), so score just that
    union (≤ 400 items) and rank it. Ranks are the item's position in
    the ORIGINAL score-ordered list, recovered via scratch arrays the
    caller allocates once (n_items int32) — not per call.

    O(topk) scatter + O(cand log cand) sort per user instead of
    O(n_items log n_items).
    """
    a_top = np.asarray(ranked_a[:topk])
    b_top = np.asarray(ranked_b[:topk])
    cand = np.unique(np.concatenate([a_top, b_top]))
    if rank_scratch_a is None or rank_scratch_b is None:
        # caller passed none/partial scratch — allocate both fresh (slow path)
        n = int(max(a_top.max(), b_top.max())) + 1
        rank_scratch_a = np.empty(n, dtype=np.int32)
        rank_scratch_b = np.empty(n, dtype=np.int32)
    # rank of each candidate in each list; topk = "not in list"
    rank_scratch_a.fill(topk)
    rank_scratch_a[a_top] = np.arange(len(a_top), dtype=np.int32)
    rank_scratch_b.fill(topk)
    rank_scratch_b[b_top] = np.arange(len(b_top), dtype=np.int32)
    ra = rank_scratch_a[cand]
    rb = rank_scratch_b[cand]
    scores = w / (RRF_K + ra + 1) + (1.0 - w) / (RRF_K + rb + 1)
    order = np.argsort(-scores)[:k]
    return cand[order]


def _build_item_knn(offsets, data, n_items, topk=20, per_user_cap=100):
    """Item-item kNN from sampled co-occurrence (review flaw #6).

    Vectorized pair-count approach: per user, the top-`per_user_cap`
    unique items by in-user frequency contribute all within-user pairs
    (a<b canonical). Pairs are deduplicated with counts via np.unique,
    then per item the top-`topk` co-occurrence counts are kept and
    cosine-normalized: sim(a,b) = co(a,b) / sqrt(c_a * c_b).

    Memory: 5000 users * 100^2/2 = 25M raw pairs (~300 MB peak int32),
    deduped to distinct pairs. Output: n_items * topk * 8B ≈ 450 MB.
    """
    log("  building item-item kNN graph (sampled co-occurrence)")
    pair_parts = []
    n_users = len(offsets) - 1
    for i in range(n_users):
        seq = data[offsets[i]:offsets[i + 1]]
        uniq, cnt = np.unique(seq, return_counts=True)
        if len(uniq) > per_user_cap:
            keep = np.argsort(-cnt)[:per_user_cap]
            uniq = uniq[keep]
        if len(uniq) < 2:
            continue
        # all within-user pairs, canonical a<b — vectorized via broadcasting
        aa, bb = np.triu_indices(len(uniq), k=1)
        lo = np.minimum(uniq[aa], uniq[bb]).astype(np.int32)
        hi = np.maximum(uniq[aa], uniq[bb]).astype(np.int32)
        pair_parts.append(np.stack([lo, hi], axis=1))
    if not pair_parts:
        return (np.full((n_items, topk), -1, dtype=np.int32),
                np.zeros((n_items, topk), dtype=np.float32))
    pairs = np.concatenate(pair_parts)
    del pair_parts
    uniq_pairs, co_counts = np.unique(pairs, axis=0, return_counts=True)
    del pairs
    log(f"  kNN: {len(uniq_pairs):,} distinct co-occurring pairs")
    # both directions: (a, b, c) and (b, a, c)
    a_col = np.concatenate([uniq_pairs[:, 0], uniq_pairs[:, 1]])
    b_col = np.concatenate([uniq_pairs[:, 1], uniq_pairs[:, 0]])
    c_col = np.concatenate([co_counts, co_counts]).astype(np.float64)
    del uniq_pairs, co_counts
    # cosine denominators from full-history item counts
    item_counts = np.bincount(data, minlength=n_items).astype(np.float64)
    denom = np.sqrt(item_counts[a_col] * np.maximum(item_counts[b_col], 1.0))
    sims = c_col / np.maximum(denom, 1e-12)
    # top-`topk` per item: sort by (item asc, sim desc)
    order = np.lexsort((-sims, a_col))
    a_col, b_col, sims = a_col[order], b_col[order], sims[order]
    del order
    # group boundaries per item
    starts = np.searchsorted(a_col, np.arange(n_items), side="left")
    ends = np.searchsorted(a_col, np.arange(n_items), side="right")
    nb_ids = np.full((n_items, topk), -1, dtype=np.int32)
    nb_sim = np.zeros((n_items, topk), dtype=np.float32)
    items_with_nb = np.where(ends > starts)[0]
    for it in items_with_nb.tolist():
        s, e = starts[it], ends[it]
        k = min(topk, e - s)
        nb_ids[it, :k] = b_col[s:s + k]
        nb_sim[it, :k] = sims[s:s + k]
    log("  item-item kNN graph built")
    return nb_ids, nb_sim


def _score_item_knn(seq, n_items, nb_ids, nb_sim):
    """Item-item kNN scorer: freq-weighted sum of neighbor sims."""
    uniq, cnt = np.unique(seq, return_counts=True)
    scores = np.zeros(n_items, dtype=np.float32)
    for h, c in zip(uniq.tolist(), cnt.tolist()):
        ids = nb_ids[h]
        valid = ids >= 0
        if valid.any():
            np.add.at(scores, ids[valid], nb_sim[h][valid] * c)
    return scores


def evaluate(embeddings, offsets, data, ts_data, user_ids, user_index,
             train_counts, user_test, n_items, ranker_model=None, seed=SEED,
             bpr_user_factors=None, bpr_item_factors=None,
             bpr_user_index=None, train_counts_cohort=None,
             user_val=None, val_start=None, item_nb=None,
             universe_mask=None, train_counts_global=None):
    """Validation-based selection, then test evaluation (v5 protocol).

    Selection pass: decay half-life + fusion weight are selected on the
    VALIDATION split, scored from train-only history — no test information
    participates in any hyperparameter choice.

    Test pass: every scorer is evaluated on TEST items using train+val
    history (the deployed task: predict the future from everything observed
    so far). Scorers: user_frequency, decay family + functional forms,
    item2vec + ablations (unique-mean, recency-weighted, windows 10/100),
    item_item_knn (if graph provided), popularity, popularity_train,
    popularity_train_global (if global train counts given), ranker, bpr_mf,
    rrf_fusion (selected w).

    v5: universe_mask (bool array over V1 ids) restricts ALL candidate
    ranking to the train-only universe; test items outside the universe
    are excluded from metrics, and universe coverage is reported.

    Also accumulates per-user rows (incl. history saturation) and the v3
    analyses; discovery is decomposed into globally-known vs cold items.
    """
    log("Evaluation")
    norm_emb = embeddings  # already normalized in-place by caller

    pop_rank = np.argsort(-train_counts)
    pop_pos = np.empty(n_items, dtype=np.int64)
    pop_pos[pop_rank] = np.arange(n_items)
    train_counts_f64 = train_counts.astype(np.float64)  # hoisted: was allocated per-user (22MB x 5000)

    # v5: train-only universe — mask popularity ranking lists too
    if universe_mask is not None:
        n_uni = int(universe_mask.sum())
        log(f"  train-only universe: {n_uni:,} of {n_items:,} items "
            f"({100.0 * n_uni / n_items:.2f}%)")
        pop_rank = pop_rank[universe_mask[pop_rank]]
        pop_pos = np.full(n_items, n_items, dtype=np.int64)
        pop_pos[pop_rank] = np.arange(len(pop_rank))

    # Cohort (v4: already selected pre-load by select_cohort; re-derive from
    # the filtered CSR and assert eligibility holds — train-ONLY >= 10)
    eligible = [int(u) for i, u in enumerate(user_ids.tolist())
                if ((val_start[i] - offsets[i]) if val_start is not None
                    else (offsets[i + 1] - offsets[i])) >= 10
                and user_test.get(u)]
    eval_users = eligible
    log(f"  cohort: {len(eval_users)} users (of {len(eligible)} eligible in filtered load)")

    # ── Pass 1: individual scorers ──
    individual_names = (["user_frequency"] +
                        [f"decay_{hl}d" for hl in DECAY_HALF_LIVES] +
                        ["decay_power1", "decay_linear365"] +
                        [f"decay_step_{w}d" for w in STEP_WINDOWS] +
                        ["item2vec", "item2vec_unique", "item2vec_recency",
                         "item2vec_w10", "item2vec_w100",
                         "popularity", "popularity_train"])
    if train_counts_global is not None:
        individual_names.append("popularity_train_global")
    if item_nb is not None:
        individual_names.append("item_item_knn")
    if ranker_model is not None:
        individual_names.append("ranker")
    if bpr_user_factors is not None:
        individual_names.append("bpr_mf")

    # Calibration accumulators (v4): per-scorer top-20 (pred, obs) pairs
    calib = {m: {"pred": [], "obs": []} for m in individual_names}

    # v4: discovery decomposition buckets (#4) — globally-known vs cold
    _SPLITS = ["overall", "repeat", "discovery",
               "discovery_known", "discovery_cold"]
    results = {m: {s: [] for s in _SPLITS}
               for m in individual_names + ["rrf_fusion"]}

    # v4: known-item mask (globally-known = has a V1 vocab entry, i.e.
    # appeared >= MIN_COUNT times in ALL events). Items outside the V1
    # vocab are cold: no embedding row, no popularity count.
    known_mask = train_counts > 0

    # v5: universe-restricted test sets — test items outside the train-only
    # universe cannot be recommended by ANY scorer, so they are excluded
    # from metrics (and their fraction is reported as universe coverage).
    if universe_mask is not None:
        n_all_test = sum(len(s) for s in user_test.values())
        n_in_uni = sum(int(universe_mask[np.fromiter(s, np.int64)].sum())
                      for s in user_test.values() if s)
        log(f"  universe coverage of test items: {n_in_uni:,}/{n_all_test:,} "
            f"({100.0 * n_in_uni / max(n_all_test, 1):.2f}%)")
        user_test = {u: {it for it in s if universe_mask[it]}
                     for u, s in user_test.items()}
        if user_val is not None:
            user_val = {u: {it for it in s if universe_mask[it]}
                       for u, s in user_val.items()}

    # popularity_train: cohort-train-only counts (#46). train_counts passed
    # in IS the cohort-train bincount in V1 space (run_recency_eval recomputes
    # it); the all-events popularity uses pop_counts_v1 (train_counts param).
    # NOTE: evaluate() receives pop_counts_v1 as train_counts (all-events);
    # the caller passes cohort-train counts separately as train_counts_cohort.
    pop_train_rank = None
    pop_train_counts_f64 = None
    if train_counts_cohort is not None:
        pop_train_rank = np.argsort(-train_counts_cohort)
        if universe_mask is not None:
            pop_train_rank = pop_train_rank[
                universe_mask[pop_train_rank]]
        pop_train_counts_f64 = train_counts_cohort.astype(np.float64)

    # v5: popularity_train_global — train-split counts over ALL users
    # (the new PRIMARY popularity baseline; no test-window leakage, no
    # cohort self-contamination).
    pop_train_global_rank = None
    pop_train_global_counts_f64 = None
    if train_counts_global is not None:
        pop_train_global_rank = np.argsort(-train_counts_global)
        pop_train_global_counts_f64 = train_counts_global.astype(np.float64)
        if universe_mask is not None:
            pop_train_global_rank = pop_train_global_rank[
                universe_mask[pop_train_global_rank]]

    # ── v3 per-user storage (#1/#2/#9/#11/#12/#14) ──
    # top-20 per scorer for coverage/repetition/tail-share: 17 scorers ×
    # 5000 users × 20 int32 ≈ 27 MB — fine.
    per_user_top20 = {m: [] for m in individual_names + ["rrf_fusion"]}
    per_user_top20_users = {m: [] for m in per_user_top20}  # uid per entry
    per_user_rows = []   # per-user headline metrics for parquet (#1)
    user_activity = {}    # uid -> (segment, n_train) (#9)
    user_test_sets = {}  # uid -> (test_items, repeat_items, discovery_items)

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

    # ═════════════════════════════════════════════════════════════════
    # v4 SELECTION PASS — pick best_hl and fusion w on VALIDATION only
    # (review flaw #1: no test information in any hyperparameter choice).
    # Per user: score decay (each hl) + ranker from TRAIN-ONLY history,
    # evaluate on VAL items. Two-stage: (1) best_hl by VAL repeat R@100,
    # (2) best w by VAL overall R@100 from cached VAL top-200s.
    # ═════════════════════════════════════════════════════════════════
    best_hl = DECAY_HALF_LIVES[1] if len(DECAY_HALF_LIVES) > 1 else DECAY_HALF_LIVES[0]
    best_w = 0.5
    # Reusable rank-scratch arrays for _fused_top_k (n_items int32 x2).
    # Allocated ONCE here; without them every call would allocate 2x n_items
    # arrays (55K calls in selection+sweep = allocation storm).
    rank_scratch_a = np.empty(n_items, dtype=np.int32)
    rank_scratch_b = np.empty(n_items, dtype=np.int32)
    if user_val is not None and val_start is not None:
        log("  selection pass: validation split")
        val_r100 = {hl: [] for hl in DECAY_HALF_LIVES}   # repeat R@100 per hl
        val_ranker_top = {}    # uid -> top-200 (train-only ctx)
        val_decay_top = {}    # uid -> {hl: top-200}
        val_sets = {}         # uid -> (val_items, val_repeat_items)
        for u in eval_users:
            i = user_index[u]
            vs = val_start[i]
            seq_tr = data[offsets[i]:vs]
            if not len(seq_tr):
                continue
            ts_tr = ts_data[offsets[i]:vs]
            v_items = user_val.get(u)
            if not v_items:
                continue
            tr_set = set(np.unique(seq_tr).tolist())
            v_repeat = v_items & tr_set
            val_sets[u] = (v_items, v_repeat)
            d_ranks = {}
            for hl in DECAY_HALF_LIVES:
                scores = _score_decay(seq_tr, ts_tr, n_items, hl)
                ranked = _top_k(scores, 100, mask=universe_mask)
                d_ranks[hl] = ranked
                if v_repeat:
                    m = _compute_metrics(ranked, v_repeat)
                    val_r100[hl].append(m["recall@100"])
            val_decay_top[u] = {hl: r[:FUSION_TOPK] for hl, r in d_ranks.items()}
            if ranker_model is not None:
                ctx_tr = seq_tr[-EVAL_CTX_LEN:]
                scores_r = _score_ranker(ranker_model, ctx_tr, all_emb_t)
                val_ranker_top[u] = _top_k(scores_r, FUSION_TOPK, mask=universe_mask)
        # stage 1: best_hl by mean VAL repeat R@100
        hl_scores = {hl: float(np.mean(v)) for hl, v in val_r100.items() if v}
        if hl_scores:
            best_hl = max(hl_scores, key=lambda h: hl_scores[h])
        log(f"  VAL-selected best decay: decay_{best_hl}d "
            f"(VAL repeat R@100 = {hl_scores.get(best_hl, float('nan')):.6f})")
        # stage 2: best w by mean VAL overall R@100 (needs ranker)
        if ranker_model is not None and val_sets:
            w_scores = {}
            for w in FUSION_SWEEP_WS:
                per_user = []
                for u, (v_items, _) in val_sets.items():
                    if u not in val_ranker_top or u not in val_decay_top:
                        continue
                    fused = _fused_top_k(val_ranker_top[u],
                                         val_decay_top[u][best_hl], w, 100,
                                         rank_scratch_a=rank_scratch_a,
                                         rank_scratch_b=rank_scratch_b)
                    per_user.append(_compute_metrics(fused, v_items)["recall@100"])
                if per_user:
                    w_scores[w] = float(np.mean(per_user))
            if w_scores:
                best_w = max(w_scores, key=lambda x: w_scores[x])
            log(f"  VAL-selected fusion weight: w={best_w:.1f} "
                f"(VAL overall R@100 = {w_scores.get(best_w, float('nan')):.6f})")
        del val_ranker_top, val_decay_top, val_sets, val_r100
    else:
        log("  WARNING: no validation split provided — hyperparameters "
            "fall back to defaults (decay_90d, w=0.5); NOT test-selected")

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
        user_top20 = {}  # scorer -> top-20 int32 (v3 per-user storage)

        # ── user_frequency ──
        ranked_uf = _score_user_frequency(seq, n_items, pop_rank, pop_pos)
        _add_metrics(results["user_frequency"], ranked_uf, test_items,
                     repeat_items, discovery_items, pop_pos, norm_emb,
                     n_items, known_mask=known_mask)
        _calib_update(calib["user_frequency"], ranked_uf,
                      _uf_scores(seq, n_items, pop_pos), test_items)
        user_top20["user_frequency"] = ranked_uf[:20].astype(np.int32)

        # ── decay scorers ──
        decay_ranks = {}
        for hl in DECAY_HALF_LIVES:
            name = f"decay_{hl}d"
            scores = _score_decay(seq, ts, n_items, hl)
            ranked = _top_k(scores, FUSION_TOPK, mask=universe_mask)
            decay_ranks[hl] = ranked
            _add_metrics(results[name], ranked, test_items,
                         repeat_items, discovery_items, pop_pos, norm_emb,
                         n_items, known_mask=known_mask)
            _calib_update(calib[name], ranked, scores, test_items)
            user_top20[name] = ranked[:20].astype(np.int32)
        user_decay_ranks[u] = decay_ranks

        # ── decay functional forms (#6) ──
        for name, scores in (
            ("decay_power1", _score_decay_power(seq, ts, n_items, 1.0)),
            ("decay_linear365", _score_decay_linear(seq, ts, n_items, 365.0)),
            *[(f"decay_step_{w}d", _score_decay_step(seq, ts, n_items, w))
              for w in STEP_WINDOWS],
        ):
            ranked = _top_k(scores, FUSION_TOPK, mask=universe_mask)
            _add_metrics(results[name], ranked, test_items,
                         repeat_items, discovery_items, pop_pos, norm_emb,
                         n_items, known_mask=known_mask)
            _calib_update(calib[name], ranked, scores, test_items)
            user_top20[name] = ranked[:20].astype(np.int32)

        # ── item2vec + ablations (#5) ──
        ctx_ts = ts[-EVAL_CTX_LEN:]
        ref_ts = int(ts[-1])
        for name, scores_i2v in (
            ("item2vec", _score_item2vec(norm_emb, ctx_items)),
            ("item2vec_unique", _score_item2vec_unique(norm_emb, ctx_items)),
            ("item2vec_recency", _score_item2vec_recency(
                norm_emb, ctx_items, ctx_ts, ref_ts)),
            ("item2vec_w10", _score_item2vec_window(norm_emb, seq, 10)),
            ("item2vec_w100", _score_item2vec_window(norm_emb, seq, 100)),
        ):
            if scores_i2v is None:
                continue
            ranked_i2v = _top_k(scores_i2v, FUSION_TOPK, mask=universe_mask)
            _add_metrics(results[name], ranked_i2v, test_items,
                         repeat_items, discovery_items, pop_pos, norm_emb,
                         n_items, known_mask=known_mask)
            _calib_update(calib[name], ranked_i2v, scores_i2v, test_items)
            user_top20[name] = ranked_i2v[:20].astype(np.int32)

        # ── item_item_knn (#6) ──
        if item_nb is not None:
            nb_ids, nb_sim = item_nb
            scores_knn = _score_item_knn(seq, n_items, nb_ids, nb_sim)
            ranked_knn = _top_k(scores_knn, FUSION_TOPK, mask=universe_mask)
            _add_metrics(results["item_item_knn"], ranked_knn, test_items,
                         repeat_items, discovery_items, pop_pos, norm_emb,
                         n_items, known_mask=known_mask)
            _calib_update(calib["item_item_knn"], ranked_knn, scores_knn,
                          test_items)
            user_top20["item_item_knn"] = ranked_knn[:20].astype(np.int32)

        # ── popularity (all-events counts; leakage sensitivity variant) ──
        _add_metrics(results["popularity"], pop_rank, test_items,
                     repeat_items, discovery_items, pop_pos, norm_emb,
                     n_items, known_mask=known_mask)
        _calib_update(calib["popularity"], pop_rank,
                      train_counts_f64, test_items)
        user_top20["popularity"] = pop_rank[:20].astype(np.int32)

        # ── popularity_train (cohort-train-only counts; leakage diagnostic) ──
        if pop_train_rank is not None and pop_train_counts_f64 is not None:
            _add_metrics(results["popularity_train"], pop_train_rank, test_items,
                         repeat_items, discovery_items, pop_pos, norm_emb,
                         n_items, known_mask=known_mask)
            _calib_update(calib["popularity_train"], pop_train_rank,
                          pop_train_counts_f64, test_items)
            user_top20["popularity_train"] = pop_train_rank[:20].astype(np.int32)

        # ── popularity_train_global (train-split counts over ALL users;
        #    v5 PRIMARY popularity baseline — no test-window leakage, no
        #    cohort self-contamination) ──
        if pop_train_global_rank is not None:
            _add_metrics(results["popularity_train_global"],
                         pop_train_global_rank, test_items,
                         repeat_items, discovery_items, pop_pos, norm_emb,
                         n_items, known_mask=known_mask)
            _calib_update(calib["popularity_train_global"],
                          pop_train_global_rank,
                          pop_train_global_counts_f64, test_items)
            user_top20["popularity_train_global"] = \
                pop_train_global_rank[:20].astype(np.int32)

        # ── ranker (GRU) ──
        if ranker_model is not None:
            scores_r = _score_ranker(ranker_model, ctx_items, all_emb_t)
            ranked_ranker = _top_k(scores_r, FUSION_TOPK, mask=universe_mask)
            user_ranker_ranks[u] = ranked_ranker
            _add_metrics(results["ranker"], ranked_ranker, test_items,
                         repeat_items, discovery_items, pop_pos, norm_emb,
                         n_items, known_mask=known_mask)
            _calib_update(calib["ranker"], ranked_ranker, scores_r, test_items)
            user_top20["ranker"] = ranked_ranker[:20].astype(np.int32)

        # ── bpr_mf ──
        if bpr_user_factors is not None and bpr_user_index is not None:
            scores_bpr = bpr_item_factors @ bpr_user_factors[bpr_user_index[u]]
            ranked_bpr = _top_k(scores_bpr, FUSION_TOPK, mask=universe_mask)
            _add_metrics(results["bpr_mf"], ranked_bpr, test_items,
                         repeat_items, discovery_items, pop_pos, norm_emb,
                         n_items, known_mask=known_mask)
            _calib_update(calib["bpr_mf"], ranked_bpr, scores_bpr, test_items)
            user_top20["bpr_mf"] = ranked_bpr[:20].astype(np.int32)

        # ── v3: per-user storage for analyses (#1/#2/#9/#11/#12/#14) ──
        n_train = int(offsets[i + 1] - offsets[i])
        user_activity[u] = (_activity_segment(n_train), n_train)
        user_test_sets[u] = (test_items, repeat_items, discovery_items)
        for m, t20 in user_top20.items():
            per_user_top20[m].append(t20)
            per_user_top20_users[m].append(u)
        # per-user headline row (#1): decay_90d as reference scorer
        row = {"user": int(u),
               "activity": user_activity[u][0],
               "n_train": n_train}
        ref_ranked = decay_ranks.get(90)
        if ref_ranked is not None:
            m90 = _compute_metrics(ref_ranked, test_items)
            row.update({f"decay90_{k}": v for k, v in m90.items()})
            # history saturation (#4 support): share of top-20 that the
            # user had already played (train+val history)
            row["decay90_saturation@20"] = float(
                len(set(ref_ranked[:20].tolist()) & train_set) / 20.0)
        # v5: per-user recall@100 for the §6.8 primary scorers, per split —
        # stored directly from the metric dicts already computed this pass
        for m in ("user_frequency", "decay_90d", "item2vec_w10", "bpr_mf",
                  "rrf_fusion", "popularity_train_global", "popularity"):
            if m == "rrf_fusion":
                continue  # computed in pass 2; appended there
            rows_m = results.get(m, {}).get("overall", [])
            if rows_m:
                row[f"{m}_overall_recall@100"] = rows_m[-1]["recall@100"]
            rows_r = results.get(m, {}).get("repeat", [])
            if rows_r:
                row[f"{m}_repeat_recall@100"] = rows_r[-1]["recall@100"]
            rows_d = results.get(m, {}).get("discovery", [])
            if rows_d:
                row[f"{m}_discovery_recall@100"] = rows_d[-1]["recall@100"]
        per_user_rows.append(row)

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

    # ── Pass 2: fusion scorers (only if ranker available) ──
    # v4: rrf_fusion uses the VAL-selected weight (best_w) — the deployed
    # configuration; the full weight sweep stays on TEST for reporting (#16).
    # NOTE: best_hl/best_w were selected on VALIDATION before the test loop.
    log(f"  fusion: decay_{best_hl}d + ranker, w={best_w:.1f} (VAL-selected)")
    fusion_sweep_rows = []  # per-user per-w metrics for the sweep
    rrf_per_user = {}       # v5: uid -> per-split recall@100 (row merge)
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

            # ── rrf_fusion (VAL-selected w) ──
            ranked_rrf = _fused_top_k(ranked_ranker, ranked_decay, best_w,
                                       100, topk=FUSION_TOPK,
                                       rank_scratch_a=rank_scratch_a,
                                       rank_scratch_b=rank_scratch_b)
            _add_metrics(results["rrf_fusion"], ranked_rrf, test_items,
                         repeat_items, discovery_items, pop_pos, norm_emb,
                         n_items, known_mask=known_mask)
            per_user_top20["rrf_fusion"].append(ranked_rrf[:20].astype(np.int32))
            per_user_top20_users["rrf_fusion"].append(u)
            # v5: rrf_fusion per-user recall@100 per split (for the row merge)
            rrf_per_user[u] = {
                "overall": _compute_metrics(ranked_rrf, test_items)["recall@100"],
                "repeat": (_compute_metrics(ranked_rrf, repeat_items)["recall@100"]
                          if repeat_items else None),
                "discovery": (_compute_metrics(ranked_rrf, discovery_items)["recall@100"]
                              if discovery_items else None),
            }

            # ── fusion weight sweep (#16, TEST reporting only) ──
            sweep = {"user": int(u)}
            for w in FUSION_SWEEP_WS:
                ranked_w = _fused_top_k(ranked_ranker, ranked_decay, w,
                                        100, topk=FUSION_TOPK,
                                        rank_scratch_a=rank_scratch_a,
                                        rank_scratch_b=rank_scratch_b)
                mw = _compute_metrics(ranked_w, test_items)
                sweep[f"w{w:.1f}_recall@20"] = mw["recall@20"]
                sweep[f"w{w:.1f}_recall@100"] = mw["recall@100"]
                if discovery_items:
                    md = _compute_metrics(ranked_w, discovery_items)
                    sweep[f"w{w:.1f}_disc_recall@100"] = md["recall@100"]
            fusion_sweep_rows.append(sweep)

    # ── Calibration (v4): reliability + ECE per scorer ──
    out = {m: _aggregate(v) for m, v in results.items()}
    for m, entry in calib.items():
        if entry["pred"] and m in out:
            out[m]["calibration"] = _compute_calibration(entry)

    # v5: merge rrf_fusion per-user values into the rows (pass-2 product)
    for row in per_user_rows:
        rp = rrf_per_user.get(row["user"])
        if rp:
            if rp["overall"] is not None:
                row["rrf_fusion_overall_recall@100"] = rp["overall"]
            if rp["repeat"] is not None:
                row["rrf_fusion_repeat_recall@100"] = rp["repeat"]
            if rp["discovery"] is not None:
                row["rrf_fusion_discovery_recall@100"] = rp["discovery"]

    # ═════════════════════════════════════════════════════════════════
    # v3 analyses (master plan) — computed from per-user storage
    # ═════════════════════════════════════════════════════════════════
    analysis = {}

    # v4: record the validation-based hyperparameter selection (flaw #1)
    analysis["val_selection"] = {
        "best_decay_half_life_days": int(best_hl),
        "best_fusion_w": float(best_w),
        "selected_on": "validation" if (user_val is not None
                                        and val_start is not None) else "defaults",
    }

    # #2 percentiles of headline metrics (decay_90d reference)
    pct_keys = [k for k in per_user_rows[0] if k.startswith("decay90_")] \
        if per_user_rows else []
    analysis["per_user_percentiles"] = {
        k: _percentiles([r[k] for r in per_user_rows])
        for k in pct_keys
    }

    # #9 activity segments: per-segment decay90 means from per_user_rows
    seg_scores = {}
    for seg in ("light", "medium", "heavy"):
        seg_rows = [r for r in per_user_rows if r["activity"] == seg]
        if seg_rows:
            seg_scores[seg] = {
                "n_users": len(seg_rows),
                "mean_n_train": round(float(np.mean([r["n_train"] for r in seg_rows])), 1),
                "decay90_recall@100": round(float(np.mean(
                    [r["decay90_recall@100"] for r in seg_rows])), 6),
                "decay90_recall@20": round(float(np.mean(
                    [r["decay90_recall@20"] for r in seg_rows])), 6),
            }
    analysis["activity_segments"] = seg_scores

    # #11 long-tail: recommendation share by bucket + per-bucket recall@20
    # (from stored top-20s; recall@100 needs full ranked lists — top-20
    # share + recall@20 per bucket is the tail diagnostic)
    tail_share = {}
    tail_recall20 = {}
    for m, tops in per_user_top20.items():
        users_m = per_user_top20_users[m]
        if not tops or len(tops) != len(users_m):
            continue
        shares = {"head": 0, "mid": 0, "tail": 0}
        rec20_hits = {"head": 0, "mid": 0, "tail": 0}
        rec20_tot = {"head": 0, "mid": 0, "tail": 0}
        for arr, u in zip(tops, users_m):
            ti, _, _ = user_test_sets[u]
            for it in arr.tolist():
                b = _tail_bucket(int(it), pop_pos, n_items)
                shares[b] += 1
                rec20_tot[b] += 1
                if it in ti:
                    rec20_hits[b] += 1
        tot = sum(shares.values()) or 1
        tail_share[m] = {k: round(v / tot, 4) for k, v in shares.items()}
        tail_recall20[m] = {
            k: round(rec20_hits[k] / rec20_tot[k], 4)
            for k in rec20_tot if rec20_tot[k] > 0
        }
    analysis["tail_share_top20"] = tail_share
    analysis["tail_recall@20"] = tail_recall20

    # #12/#14 coverage + repetition share from stored top-20s
    cov_rep = {}
    for m, tops in per_user_top20.items():
        users_m = per_user_top20_users[m]
        if not tops or len(tops) != len(users_m):
            continue
        all_recs = np.concatenate(tops)
        unique_items = len(np.unique(all_recs))
        rep_share = float(np.mean([
            np.mean([1.0 if it in user_test_sets[u][1] else 0.0
                     for it in arr.tolist()])
            for arr, u in zip(tops, users_m)
        ]))
        cov_rep[m] = {
            "catalog_coverage@20": round(unique_items / n_items, 6),
            "repeat_share_top20": round(rep_share, 4),
        }
    analysis["coverage_repetition"] = cov_rep

    # #25 paired bootstrap CIs for key scorer pairs (recall@100).
    # v5: SPLIT-LABELED — each pair is bootstrapped on overall AND repeat
    # AND discovery per-user vectors, with the split named in the key
    # (fixes the §6.8 split-mismatch inconsistency: §6.2/§6.4 quote
    # repeat/discovery numbers, the old CIs were overall-only).
    pairs = [
        ("decay_90d", "user_frequency"),
        ("decay_90d", "popularity_train_global"),
        ("item2vec_w10", "popularity_train_global"),
        ("bpr_mf", "popularity_train_global"),
        ("rrf_fusion", "decay_90d"),
    ]
    boot = {}
    for a_name, b_name in pairs:
        b_key = b_name if b_name in results else "popularity"
        for split in ("overall", "repeat", "discovery"):
            rows_a = results.get(a_name, {}).get(split, [])
            rows_b = results.get(b_key, {}).get(split, [])
            if rows_a and rows_b and len(rows_a) == len(rows_b):
                boot[f"{a_name}_vs_{b_key}|{split}"] = _paired_bootstrap(
                    [r["recall@100"] for r in rows_a],
                    [r["recall@100"] for r in rows_b])
    analysis["bootstrap_cis_recall@100"] = boot

    # v5: saturation curve — per-user proportion of top-K items already in
    # the user's history, for the VAL-selected decay scorer (empirical
    # saturation mechanism; K sweep). Uses the cached top-FUSION_TOPK decay
    # ranks; Ks beyond FUSION_TOPK are clamped to it.
    sat_curve = {}
    sat_users = list(user_decay_ranks.keys())
    if sat_users:
        for K in SATURATION_KS:
            k_eff = min(K, FUSION_TOPK)
            vals = []
            for u in sat_users:
                ranked = user_decay_ranks[u].get(best_hl)
                if ranked is None:
                    continue
                i = user_index[u]
                seq = data[offsets[i]:offsets[i + 1]]
                hist = set(seq.tolist())
                vals.append(len(set(ranked[:k_eff].tolist()) & hist) / k_eff)
            if vals:
                sat_curve[f"K={K}"] = round(float(np.mean(vals)), 4)
        analysis["saturation_curve"] = {
            "scorer": f"decay_{best_hl}d",
            "seen_item_share_by_K": sat_curve,
        }

    # #16 fusion sweep aggregates
    if fusion_sweep_rows:
        sweep_agg = {}
        for w in FUSION_SWEEP_WS:
            r20 = [r[f"w{w:.1f}_recall@20"] for r in fusion_sweep_rows]
            r100 = [r[f"w{w:.1f}_recall@100"] for r in fusion_sweep_rows]
            d100 = [r.get(f"w{w:.1f}_disc_recall@100", 0.0)
                    for r in fusion_sweep_rows]
            sweep_agg[f"w{w:.1f}"] = {
                "recall@20": round(float(np.mean(r20)), 6),
                "recall@100": round(float(np.mean(r100)), 6),
                "disc_recall@100": round(float(np.mean(d100)), 6),
            }
        analysis["fusion_sweep"] = sweep_agg

    # #46 popularity leakage delta
    if "popularity_train" in out and "popularity" in out:
        pt = out["popularity_train"].get("overall", {})
        pp = out["popularity"].get("overall", {})
        analysis["popularity_train_delta"] = {
            "all_events_recall@100": pp.get("recall@100"),
            "train_only_recall@100": pt.get("recall@100"),
            "delta": round((pt.get("recall@100", 0.0)
                            - pp.get("recall@100", 0.0)), 6),
        }

    return out, analysis, per_user_rows


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
                   f"precision@{RECALL_KS[0]}", f"precision@{RECALL_KS[1]}",
                   f"ndcg@{RECALL_KS[0]}", f"ndcg@{RECALL_KS[1]}",
                   f"mrr@{MRR_K}", f"hitrate@{HITRATE_K}"]
    header = "| Model | Split | " + " | ".join(metric_keys) + \
             " | Novelty | Diversity |"
    sep = "|-------|-------|" + "|".join(["------"] * len(metric_keys)) + \
          "|---------|-----------|"
    lines = ["# Recency Evaluation Report", "",
             f"**Generated:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
             "**Kernel:** lb_eval.py", "",
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
              "- **user_frequency:** plain per-user item counts, ties by item id "
              "(v4: deterministic, no future info)",
              "- **decay_30d/90d/365d:** exponential decay over train+val events "
              "(half-life in days)",
              "- **decay_power1:** power-law decay 1/(1+age_days) (#6)",
              "- **decay_linear365:** linear decay to zero at 365 days (#6)",
              "- **decay_step_{7,30,90,365}d:** step decay — full weight inside "
              "window, zero outside (#6)",
              "- **item2vec:** mean embedding of last 50 history items, cosine sim",
              "- **item2vec_unique:** mean over unique context items (ablation)",
              "- **item2vec_recency:** decay-weighted mean embedding (ablation)",
              "- **item2vec_w10 / w100:** window 10 / 100 context means (ablations)",
              "- **item_item_knn:** co-occurrence item-item kNN (top-20 neighbors, "
              "cosine-normalized; v4 baseline)",
              "- **popularity_train_global:** train-split counts over ALL users "
              "— PRIMARY popularity baseline (v5; no test-window leakage, no "
              "cohort self-contamination)",
              "- **popularity_train:** cohort-train-only counts — leakage "
              "diagnostic (cohort self-contamination)",
              "- **popularity:** all-events counts — leakage-sensitivity variant "
              "(includes test-window events)",
              "- **ranker:** GRU sequence model (from lb-ranker)",
              "- **bpr_mf:** BPR-MF implicit-feedback baseline (from lb-bpr; "
              "V2-space factors translated to V1, user factors over all "
              "train users)",
              "- **rrf_fusion:** weighted RRF (k=60) of ranker + best decay, "
              "w and decay half-life selected on VALIDATION (v4)",
              "",
              "## Candidate universe (v5)", "",
              "All scorers rank within the TRAIN-ONLY universe (V2 trainprep",
              "vocab, train-split counts >= 10, translated to V1 space).",
              "Test items outside the universe are excluded from metrics;",
              "universe coverage is logged. The all-events V1 catalog of",
              "v4/v5-kernel outputs is retained as the fixed-catalog",
              "sensitivity variant.",
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
              "- **Context:** last 50 history (train+val) items",
              "- **Eval:** repeats NOT excluded (legitimate predictions)",
              "- **Eval type:** set-based — recall/MRR against the full "
              "test-set of held-out items, not next-item prediction",
              "- **Protocol (v4):** hyperparameters (best decay half-life, "
              "fusion weight) are selected on the VALIDATION split only; "
              "TEST evaluation uses train+val history. No test information "
              "enters any selection decision.",
              "- **Discovery decomposition (v4):** discovery split is reported "
              "as discovery_known (user-unseen but globally known items) and "
              "discovery_cold (globally unseen items).",
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
              "- **Popularity baselines (v4):** popularity_train (cohort "
              "train-only counts) is the PRIMARY baseline; popularity "
              "(all-events counts from the V1 vocab table) is retained as "
              "a leakage-sensitivity variant — its counts include "
              "test-window events."]
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
| `user_frequency` | Plain per-user item counts (ties by item id) |
| `decay_30d` | Exponential decay, half-life=30 days |
| `decay_90d` | Exponential decay, half-life=90 days |
| `decay_365d` | Exponential decay, half-life=365 days |
| `item2vec` | Mean embedding scorer |
| `item2vec_unique` / `_recency` / `_w10` / `_w100` | item2vec profile ablations (v4) |
| `item_item_knn` | Co-occurrence item-item kNN baseline (v4) |
| `popularity_train_global` | Train-split counts over ALL users — PRIMARY baseline (v5) |
| `popularity_train` | Cohort-train-only frequency — leakage diagnostic |
| `popularity` | All-events frequency — leakage-sensitivity variant |
| `ranker` | GRU sequence model |
| `bpr_mf` | BPR-MF implicit-feedback baseline (lb-bpr) |
| `rrf_fusion` | Weighted RRF; w + decay selected on VALIDATION (v4) |
| `decay_power1` / `decay_linear365` / `decay_step_*` | Decay functional forms (#6) |

## v3 analyses

Master-plan analyses (per-user percentiles, activity segments, long-tail
share, bootstrap CIs, fusion weight sweep, coverage/repetition,
popularity-train delta) are in `analysis.json`; per-user rows in
`data/per_user_metrics.parquet`.
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
                     output_dir, ranker_path=None, bpr_path=None,
                     multi_seed=False):
    """Full pipeline: V1 vocab rebuild → translate → eval → reports.

    v4.1: multi_seed=True runs the 3-seed robustness loop (cohort
    re-selected per seed; CSR re-loaded per seed — the cohort filter is
    seed-dependent) plus the global-chronological-split sensitivity
    pass, writing multiseed.json + globalsplit.json alongside the
    seed-42 primary outputs (which stay byte-comparable to v4).
    """
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
    cohort_ids, n_eligible, all_train_users = select_cohort(events_path)
    log(f"  cohort selected: {len(cohort_ids):,} of {n_eligible:,} eligible")

    # ── Step 2: Load V2 vocab + events (cohort-filtered), translate to V1 ──
    n_v2 = load_n_items(vocab_path)
    log(f"V2 vocab size: {n_v2:,}")

    v2_to_v1 = build_v2_to_v1(vocab_path, mbid_to_v1_id, n_v2)
    del mbid_to_v1_id
    gc.collect()

    (offsets, data_v2, ts_data_v2, user_ids, user_index, train_counts_v2,
     user_test_v2, user_val_v2, val_start_v2) = \
        load_csr(events_path, n_v2, cohort_users=cohort_ids)

    log(f"  cohort users loaded={len(user_ids):,} "
        f"(cohort {len(cohort_ids):,})")
    log(f"  train+val events={len(data_v2):,} (cohort-filtered)")
    log(f"  RSS {rss_mb():.0f} MB")

    # Translate everything to V1 space
    log("Translating V2 -> V1 space")
    data_v1, ts_data_v1, user_test_v1 = translate_csr_to_v1(
        data_v2, ts_data_v2, user_test_v2, v2_to_v1)
    # v4: translate val sets too (val_start is index-space, unchanged)
    user_val_v1 = {}
    for uid, items in user_val_v2.items():
        user_val_v1[int(uid)] = {int(v2_to_v1[i]) for i in items}
    del data_v2, ts_data_v2, user_test_v2, user_val_v2
    gc.collect()
    log(f"  RSS {rss_mb():.0f} MB")

    # ── BPR-MF factors (V2-space; translate to V1 before dropping the map) ──
    bpr_user_factors = None
    bpr_item_factors = None
    bpr_user_index = None
    if bpr_path and Path(bpr_path).exists():
        log(f"Loading BPR-MF factors from {bpr_path}")
        bpr_user_factors, bpr_item_factors = load_bpr(
            bpr_path, v2_to_v1, all_train_users, n_v1)
        # Row index into user_factors: all_train_users is sorted; lb-bpr's
        # CSR user order is sorted over the same user set
        bpr_user_index = {int(u): i for i, u in enumerate(all_train_users.tolist())}
        log("  BPR factors loaded (user index over all train users)")
    else:
        log("WARNING: bpr_final.pt not found, skipping bpr_mf scorer")
    del v2_to_v1
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
    # v4: item-item kNN graph from the cohort CSR (flaw #6 baseline)
    item_nb = _build_item_knn(offsets, data_v1, n_v1)
    log(f"  RSS {rss_mb():.0f} MB after kNN build")
    results, analysis, per_user_rows = evaluate(
        embeddings, offsets, data_v1, ts_data_v1, user_ids,
        user_index, pop_counts_v1, user_test_v1, n_v1,
        ranker_model, bpr_user_factors=bpr_user_factors,
        bpr_item_factors=bpr_item_factors,
        bpr_user_index=bpr_user_index,
        train_counts_cohort=train_counts_v1,
        user_val=user_val_v1, val_start=val_start_v2, item_nb=item_nb)
    del item_nb
    gc.collect()

    log("STEP 5: Writing reports")
    _write_reports(results, output_dir)

    # ── Step 6 (v3): analysis.json + per-user parquet ──
    analysis_path = output_dir / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    log(f"  wrote {analysis_path}")

    if per_user_rows:
        cols = {}
        keys = [k for k in per_user_rows[0] if k != "activity"]
        cols["user"] = pa.array([r["user"] for r in per_user_rows],
                                type=pa.int32())
        cols["activity"] = pa.array([r["activity"] for r in per_user_rows])
        for k in keys:
            if k == "user":
                continue
            cols[k] = pa.array([float(r[k]) for r in per_user_rows],
                               type=pa.float64())
        pq.write_table(pa.table(cols),
                       str(output_dir / "data" / "per_user_metrics.parquet"))
        log(f"  wrote data/per_user_metrics.parquet ({len(per_user_rows)} rows)")

    log("DONE")
    return results

    # NOTE: multi-seed + global-split machinery lives in
    # run_multiseed_eval() below; run_recency_eval stays the single-seed
    # primary path (byte-comparable to v4 outputs).


# ═══════════════════════════════════════════════════════════════════════
# v4.1 MULTI-SEED + GLOBAL-SPLIT (kernel v5)
# ═══════════════════════════════════════════════════════════════════════

def _flatten_primary(results):
    """Per-scorer headline metrics for cross-seed aggregation:
    {scorer: {split.metric: value}} for overall/repeat/discovery."""
    flat = {}
    for scorer, splits in results.items():
        row = {}
        for sp in ("overall", "repeat", "discovery"):
            m = splits.get(sp) if isinstance(splits, dict) else None
            m = m or {}
            for k, v in m.items():
                if k == "n_users" or not isinstance(v, (int, float)):
                    continue
                row[f"{sp}.{k}"] = float(v)
        flat[scorer] = row
    return flat


def _mean_sd_across(values):
    """values: list of dicts (same keys). Returns {k: {mean, sd, n}}."""
    out = {}
    if not values:
        return out
    keys = values[0].keys()
    for k in keys:
        xs = [v[k] for v in values if k in v]
        arr = np.asarray(xs, dtype=np.float64)
        out[k] = {"mean": float(arr.mean()),
                  "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                  "n": len(arr)}
    return out


def _global_split_test(events_path, cohort_ids, n_v2, cutoff_days=14):
    """Global-chronological test sets: per user, events after the GLOBAL
    cutoff (max ts across cohort train+val events minus cutoff_days).
    Returns (user_gtest dict uid->set, global_cutoff_ts)."""
    cohort_arr = cohort_ids
    pf = pq.ParquetFile(str(events_path))
    max_ts = -2**63
    # pass 1: global max ts over cohort train+val rows
    for batch in pf.iter_batches(batch_size=2_000_000,
                                 columns=["user", "split", "ts"]):
        users_np = batch.column("user").to_numpy()
        splits_np = batch.column("split").to_numpy()
        ts_np = batch.column("ts").to_numpy()
        m = np.isin(users_np, cohort_arr) & ((splits_np == 0) | (splits_np == 1))
        if m.any():
            max_ts = max(max_ts, int(ts_np[m].max()))
    cutoff = max_ts - cutoff_days * 86400
    # pass 2: collect test items after cutoff (any split — the global
    # window ignores the per-user split assignment)
    user_gtest = defaultdict(set)
    for batch in pf.iter_batches(batch_size=2_000_000,
                                 columns=["user", "item_id", "ts"]):
        users_np = batch.column("user").to_numpy()
        items_np = batch.column("item_id").to_numpy()
        ts_np = batch.column("ts").to_numpy()
        m = np.isin(users_np, cohort_arr) & (ts_np > cutoff)
        if m.any():
            for u, it in zip(users_np[m].tolist(), items_np[m].tolist()):
                user_gtest[u].add(int(it))
    return dict(user_gtest), cutoff


def run_multiseed_eval(listens_path, events_path, vocab_path, emb_path,
                       output_dir, ranker_path=None, bpr_path=None):
    """v4.1 robustness round: 3 cohort seeds + global-split sensitivity.

    Heavy one-time setup (V1 vocab, V2 map, embeddings, ranker, BPR) is
    done once; per seed only cohort selection + cohort-filtered CSR load
    + kNN build + evaluate() re-run. Primary seed-42 outputs are written
    exactly as in v4 (metrics.json etc.); robustness outputs go to
    multiseed.json + globalsplit.json.
    """
    output_dir = Path(output_dir)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)

    # ── one-time setup (identical to run_recency_eval steps 1-3) ──
    mbid_to_v1_id, v1_table = build_v1_vocab(listens_path)
    n_v1 = len(mbid_to_v1_id)
    log(f"  V1 vocab: {n_v1:,} items")

    pop_counts_v1 = np.zeros(n_v1, dtype=np.int64)
    _vc = v1_table.column("count").to_numpy()
    _vi = v1_table.column("v1_item_id").to_numpy()
    pop_counts_v1[_vi] = _vc
    del v1_table, _vc, _vi
    gc.collect()

    n_v2 = load_n_items(vocab_path)
    v2_to_v1 = build_v2_to_v1(vocab_path, mbid_to_v1_id, n_v2)
    del mbid_to_v1_id
    gc.collect()

    # v5: train-only candidate universe — the V2 trainprep vocab (train-split
    # counts >= 10) translated into V1 space. Items outside it cannot be
    # recommended (they entered the all-events V1 catalog via future
    # interactions — the leakage review-2 flagged).
    universe_mask = np.zeros(n_v1, dtype=bool)
    universe_mask[v2_to_v1] = True
    log(f"  train-only universe: {int(universe_mask.sum()):,} of {n_v1:,} V1 items")

    # v5: global train-split counts (ALL users, train rows only) in V1
    # space — the basis of popularity_train_global (PRIMARY baseline).
    train_counts_global_v2 = np.zeros(n_v2, dtype=np.int64)
    _pf = pq.ParquetFile(str(events_path))
    for _b in _pf.iter_batches(batch_size=2_000_000,
                               columns=["item_id", "split"]):
        _m = _b.column("split").to_numpy() == 0
        if _m.any():
            np.add.at(train_counts_global_v2,
                      _b.column("item_id").to_numpy()[_m], 1)
    train_counts_global = np.zeros(n_v1, dtype=np.int64)
    np.add.at(train_counts_global, v2_to_v1, train_counts_global_v2)
    del train_counts_global_v2, _pf
    gc.collect()
    log(f"  global train counts: {int((train_counts_global > 0).sum()):,} "
        f"items with >=1 train event")

    embeddings = load_embeddings(emb_path, n_v1)
    embeddings /= (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    log(f"  embeddings loaded+normalized: {embeddings.shape}")

    ranker_model = None
    if ranker_path and Path(ranker_path).exists():
        ranker_model = GRURanker(n_v1, EMB_DIM, GRU_HIDDEN, GRU_LAYERS, "cpu")
        ranker_model.load(ranker_path)
        log("  ranker loaded")

    # ── seed loop ──
    bpr_user_factors = bpr_item_factors = bpr_user_index = None
    s42 = None
    per_seed_flat = {}
    seed42_outputs = None
    for seed in MULTI_SEEDS:
        log(f"SEED {seed}: cohort + CSR + evaluate")
        cohort_ids, n_eligible, all_train_users = select_cohort(
            events_path, seed=seed)

        (offsets, data_v2, ts_data_v2, user_ids, user_index, train_counts_v2,
         user_test_v2, user_val_v2, val_start_v2) = load_csr(
            events_path, n_v2, cohort_users=cohort_ids)

        data_v1, ts_data_v1, user_test_v1 = translate_csr_to_v1(
            data_v2, ts_data_v2, user_test_v2, v2_to_v1)
        user_val_v1 = {int(uid): {int(v2_to_v1[i]) for i in items}
                       for uid, items in user_val_v2.items()}
        del data_v2, ts_data_v2, user_test_v2, user_val_v2
        gc.collect()

        if bpr_path and Path(bpr_path).exists() and bpr_user_factors is None:
            bpr_user_factors, bpr_item_factors = load_bpr(
                bpr_path, v2_to_v1, all_train_users, n_v1)
            bpr_user_index = {int(u): i
                              for i, u in enumerate(all_train_users.tolist())}

        train_counts_v1 = np.bincount(data_v1, minlength=n_v1).astype(np.int64)
        item_nb = _build_item_knn(offsets, data_v1, n_v1)
        results, analysis, per_user_rows = evaluate(
            embeddings, offsets, data_v1, ts_data_v1, user_ids,
            user_index, pop_counts_v1, user_test_v1, n_v1,
            ranker_model, seed=seed,
            bpr_user_factors=bpr_user_factors,
            bpr_item_factors=bpr_item_factors,
            bpr_user_index=bpr_user_index,
            train_counts_cohort=train_counts_v1,
            user_val=user_val_v1, val_start=val_start_v2, item_nb=item_nb,
            universe_mask=universe_mask,
            train_counts_global=train_counts_global)
        del item_nb
        gc.collect()

        per_seed_flat[seed] = _flatten_primary(results)
        if seed == MULTI_SEEDS[0]:
            seed42_outputs = (results, analysis, per_user_rows)
            # primary outputs (byte-comparable to v4)
            _write_reports(results, output_dir)
            (output_dir / "analysis.json").write_text(
                json.dumps(analysis, indent=2), encoding="utf-8")
            if per_user_rows:
                cols = {"user": pa.array([r["user"] for r in per_user_rows],
                                         type=pa.int32()),
                        "activity": pa.array([r["activity"] for r in per_user_rows])}
                # v5: keys can be sparse per row (e.g. rrf_fusion_* only
                # when a user has that split) — union of keys, NaN fill
                all_keys = []
                for r in per_user_rows:
                    for k in r:
                        if k not in all_keys and k not in ("user", "activity"):
                            all_keys.append(k)
                for k in all_keys:
                    cols[k] = pa.array(
                        [float(r[k]) if k in r else None
                         for r in per_user_rows],
                        type=pa.float64())
                pq.write_table(pa.table(cols),
                               str(output_dir / "data" / "per_user_metrics.parquet"))

        # keep seed-42 CSR for the global-split pass
        if seed == MULTI_SEEDS[0]:
            s42 = (offsets, data_v1, ts_data_v1, user_ids, user_index,
                   user_val_v1, val_start_v2, cohort_ids)
        else:
            del offsets, data_v1, ts_data_v1, user_ids, user_index
            gc.collect()

    # ── multiseed.json ──
    multiseed = {str(s): per_seed_flat[s] for s in per_seed_flat}
    agg = {scorer: _mean_sd_across([per_seed_flat[s][scorer]
                                    for s in MULTI_SEEDS
                                    if scorer in per_seed_flat[s]])
           for scorer in per_seed_flat[MULTI_SEEDS[0]]}
    multiseed["aggregate_mean_sd"] = agg
    (output_dir / "multiseed.json").write_text(
        json.dumps(multiseed, indent=2), encoding="utf-8")
    log(f"  wrote multiseed.json ({len(MULTI_SEEDS)} seeds)")

    # ── global-chronological-split sensitivity (seed-42 cohort) ──
    log("GLOBAL-SPLIT sensitivity pass")
    if s42 is None:
        sys.exit("FATAL: seed-42 CSR not retained for global-split pass")
    (offsets, data_v1, ts_data_v1, user_ids, user_index,
     user_val_v1, val_start_v2, cohort_ids) = s42
    user_gtest_v2, cutoff = _global_split_test(events_path, cohort_ids, n_v2)
    user_gtest_v1 = {int(u): {int(v2_to_v1[i]) for i in items}
                    for u, items in user_gtest_v2.items()}
    del user_gtest_v2
    log(f"  global cutoff ts={cutoff:,}; users with post-cutoff events: "
        f"{len(user_gtest_v1):,}")

    # Sensitivity: swap test sets, keep VAL-selected hyperparams from the
    # primary run (re-selection would need a global VAL too — noted in
    # report; hyperparams are frozen here by design).
    gs_results, _, _ = evaluate(
        embeddings, offsets, data_v1, ts_data_v1, user_ids,
        user_index, pop_counts_v1, user_gtest_v1, n_v1,
        ranker_model, seed=MULTI_SEEDS[0],
        bpr_user_factors=bpr_user_factors,
        bpr_item_factors=bpr_item_factors,
        bpr_user_index=bpr_user_index,
        train_counts_cohort=np.bincount(data_v1, minlength=n_v1).astype(np.int64),
        user_val=user_val_v1, val_start=val_start_v2,
        item_nb=_build_item_knn(offsets, data_v1, n_v1),
        universe_mask=universe_mask,
        train_counts_global=train_counts_global)
    gs_flat = _flatten_primary(gs_results)
    gs_out = {"global_cutoff_ts": int(cutoff),
              "cutoff_days": 14,
              "note": ("sensitivity check, not a second benchmark; "
                       "hyperparams frozen from per-user-split VAL selection"),
              "scorers": gs_flat}
    (output_dir / "globalsplit.json").write_text(
        json.dumps(gs_out, indent=2), encoding="utf-8")
    log(f"  wrote globalsplit.json")

    log("MULTISEED DONE")
    if seed42_outputs is None:
        sys.exit("FATAL: primary seed outputs missing")
    return seed42_outputs[0]


def main():
    """Kaggle entry point — discover inputs, run full pipeline."""
    root = Path("/kaggle/input")
    listens_files = list(root.rglob("listens.parquet"))
    events_files = list(root.rglob("events.parquet"))
    vocab_files = list(root.rglob("vocab.parquet"))
    emb_files = list(root.rglob("item2vec_final.npy"))
    ranker_files = list(root.rglob("ranker_final.pt"))
    bpr_files = list(root.rglob("bpr_final.pt"))

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
    if bpr_files:
        log(f"BPR:    {bpr_files[0]}")
    else:
        log("BPR:    NOT FOUND (bpr_mf scorer will be skipped)")

    run_multiseed_eval(listens_files[0], events_files[0], vocab_files[0],
                       emb_files[0], Path("/kaggle/working"),
                       ranker_path=str(ranker_files[0]) if ranker_files else None,
                       bpr_path=str(bpr_files[0]) if bpr_files else None)


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
        # Mark each user's LAST event as test (split=2) and SECOND-TO-LAST
        # as validation (split=1) so select_cohort has eligible users
        # (>=10 train AND >=1 test) and the v4 VAL-selection path runs.
        last_idx_by_user = {}
        second_last_idx_by_user = {}
        seen = {}
        for i, r in enumerate(events_rows):
            u_ = r[0]
            if u_ in seen:
                second_last_idx_by_user[u_] = seen[u_]
            seen[u_] = i
        last_idx_by_user = dict(seen)
        events_rows = [
            (u, ts, it,
             2 if i in last_idx_by_user.values()
             else (1 if i in second_last_idx_by_user.values() else s))
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

        # ── 8b. Test load_bpr (V2-space factors -> V1 translation) ──
        import torch as _torch
        # Fake BPR checkpoint: user factors over ALL 8 users (sorted),
        # item factors in V2 space (28 rows)
        fake_uf = rng.standard_normal((n_users, 64)).astype(np.float32)
        fake_itf_v2 = rng.standard_normal((n_v2, 64)).astype(np.float32)
        bpr_ckpt = {
            "user_factors.weight": _torch.from_numpy(fake_uf),
            "item_factors.weight": _torch.from_numpy(fake_itf_v2),
        }
        bpr_path_t = tmpdir / "bpr_final.pt"
        _torch.save(bpr_ckpt, str(bpr_path_t))
        # all_train_users: every user has >=1 train event here
        all_users_t = np.arange(n_users, dtype=np.int32)
        uf_t, itf_v1_t = load_bpr(str(bpr_path_t), v2_to_v1,
                                  all_users_t, n_v1_rebuilt)
        assert uf_t.shape == (n_users, 64)
        assert itf_v1_t.shape == (n_v1, 64)
        # Every V2 item's factor must appear at its V1 position
        for v2i in range(n_v2):
            v1i = int(v2_to_v1[v2i])
            assert np.allclose(itf_v1_t[v1i], fake_itf_v2[v2i]), \
                f"V2 item {v2i} factor not translated to V1 {v1i}"
        # V1-only items (excluded from V2) must be zero vectors
        v1_only = set(range(n_v1)) - {int(v) for v in v2_to_v1}
        for v1i in v1_only:
            assert not itf_v1_t[v1i].any(), f"V1-only item {v1i} should be zero"
        # User-factor lookup: user id -> row
        bpr_uix = {int(u): i for i, u in enumerate(all_users_t.tolist())}
        assert np.allclose(uf_t[bpr_uix[3]], fake_uf[3])
        log("  load_bpr: V2->V1 factor translation PASSED")

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
        cohort_ids, n_elig, all_users = select_cohort(str(events_path), eval_sample=5)
        assert n_elig == n_users, f"eligible {n_elig} != {n_users} (all users have >=10 train + 1 test)"
        assert len(cohort_ids) == min(5, n_users)
        assert list(cohort_ids) == sorted(cohort_ids), "cohort must be sorted"
        assert len(all_users) == n_users, "all_train_users must cover every user with >=1 train event"
        log(f"  select_cohort: {len(cohort_ids)} of {n_elig} eligible PASSED")

        # Cohort-filtered load_csr: only cohort users' rows
        (c_off, c_data, c_ts, c_uids, c_uix, c_tc, c_test, c_val, c_vs) = \
            load_csr(str(events_path), n_v2, cohort_users=cohort_ids)
        assert set(int(x) for x in c_uids) <= set(int(x) for x in cohort_ids), \
            "loaded users must be subset of cohort"
        assert all(len(c_test.get(int(u), set())) >= 1 for u in c_uids), \
            "every loaded cohort user must have test items"
        # v4: val structures present; val rows sit between val_start and end
        assert len(c_vs) == len(c_uids), "val_start parallel to user_ids"
        for i, u in enumerate(c_uids.tolist()):
            n_val = len(c_val.get(int(u), set()))
            assert c_vs[i] + n_val <= c_off[i + 1], \
                "val block must fit between val_start and user end"
        log("  cohort-filtered load_csr (train+val CSR, val boundary) PASSED")

        # ── 9c. Test calibration helpers (v4) ──
        calib_entry = {"pred": [], "obs": []}
        fake_ranked = np.array([3, 1, 7, 0, 5, 2, 6, 4])
        fake_scores = np.array([0.9, 0.7, 0.5, 0.3, 0.1, 0.05, 0.02, 0.0])
        _calib_update(calib_entry, fake_ranked, fake_scores, {3, 7})
        cal = _compute_calibration(calib_entry)
        assert 0.0 <= cal["ece"] <= 1.0, f"ECE out of range: {cal['ece']}"
        assert sum(b[2] for b in cal["bins"]) <= 1.0 + 1e-9, "bin weights must sum <= 1"
        log(f"  calibration: ECE={cal['ece']} PASSED")

        # ── 9d. Test v3/v4 analyses end-to-end on tiny fixtures ──
        # Translate the cohort CSR to V1 and run evaluate() with all
        # optional models, then assert analysis structure.
        tr_data, tr_ts, tr_test = translate_csr_to_v1(
            c_data, c_ts, c_test, v2_to_v1)
        # v4: translate VAL sets too
        c_val_v1 = {}
        for uid, items in c_val.items():
            c_val_v1[int(uid)] = {int(v2_to_v1[i]) for i in items}
        emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        cohort_counts = np.bincount(tr_data, minlength=n_v1).astype(np.int64)
        # all-events popularity counts in V1 space (from the vocab table)
        pop_counts_v1_t = np.zeros(n_v1, dtype=np.int64)
        pop_counts_v1_t[v1_table.column("v1_item_id").to_numpy()] = \
            v1_table.column("count").to_numpy()
        # v4: tiny item-item kNN graph from the fixture CSR
        item_nb_t = _build_item_knn(c_off, tr_data, n_v1)
        # v5: train-only universe mask — V2 items translated to V1
        uni_mask_t = np.zeros(n_v1, dtype=bool)
        uni_mask_t[v2_to_v1] = True
        # v5: global train counts (fixture: bincount over ALL fixture
        # train rows in V1 space — every V2 item is train-seen here)
        gtc_t = np.zeros(n_v1, dtype=np.int64)
        np.add.at(gtc_t, v2_to_v1,
                  np.bincount(np.arange(n_v2), minlength=n_v2))
        res_v3, ana_v3, rows_v3 = evaluate(
            emb_norm, c_off, tr_data, tr_ts, c_uids, c_uix,
            pop_counts_v1_t, tr_test, n_v1,
            ranker_model=ranker, bpr_user_factors=uf_t,
            bpr_item_factors=itf_v1_t, bpr_user_index=bpr_uix,
            train_counts_cohort=cohort_counts,
            user_val=c_val_v1, val_start=c_vs, item_nb=item_nb_t,
            universe_mask=uni_mask_t, train_counts_global=gtc_t)
        # metrics.json-compatible scorer set
        for m in ("user_frequency", "decay_90d", "decay_power1",
                  "decay_linear365", "decay_step_30d", "item2vec",
                  "item2vec_unique", "item2vec_recency",
                  "item2vec_w10", "item2vec_w100", "item_item_knn",
                  "popularity", "popularity_train",
                  "popularity_train_global", "ranker", "bpr_mf",
                  "rrf_fusion"):
            assert m in res_v3, f"missing scorer {m}"
            assert "overall" in res_v3[m] and res_v3[m]["overall"], m
        assert "w_fusion_05" not in res_v3
        # v4: VAL selection recorded + ran on validation
        assert "val_selection" in ana_v3, "missing val_selection in analysis"
        assert ana_v3["val_selection"]["selected_on"] == "validation", \
            "VAL selection did not run"
        assert ana_v3["val_selection"]["best_decay_half_life_days"] in \
            DECAY_HALF_LIVES
        # v4: discovery decomposition buckets exist
        for m in ("item2vec", "popularity_train"):
            assert "discovery_known" in res_v3[m] or \
                "discovery_cold" in res_v3[m], f"{m} missing discovery split"
        # v4: saturation stat in per-user rows
        assert all("decay90_saturation@20" in r for r in rows_v3), \
            "missing decay90_saturation@20 in per-user rows"
        # v5: split-labeled bootstrap keys (pair|split format)
        boot_keys = list(ana_v3["bootstrap_cis_recall@100"].keys())
        assert any("|overall" in k for k in boot_keys), \
            "bootstrap keys missing |overall split label"
        assert any("|repeat" in k for k in boot_keys), \
            "bootstrap keys missing |repeat split label"
        # v5: saturation curve present with K sweep
        assert "saturation_curve" in ana_v3, "missing saturation_curve"
        assert len(ana_v3["saturation_curve"]["seen_item_share_by_K"]) >= 1
        # v5: per-user rows carry primary-scorer recall@100 columns
        assert all("user_frequency_overall_recall@100" in r for r in rows_v3), \
            "missing user_frequency_overall_recall@100 in per-user rows"
        assert any("rrf_fusion_overall_recall@100" in r for r in rows_v3), \
            "missing rrf_fusion_overall_recall@100 in per-user rows"
        # analysis structure
        assert set(ana_v3["activity_segments"]) <= {"light", "medium", "heavy"}
        assert sum(v["n_users"] for v in ana_v3["activity_segments"].values()) \
            == len(rows_v3)
        for m, share in ana_v3["tail_share_top20"].items():
            assert abs(sum(share.values()) - 1.0) < 1e-6, f"{m} tail share != 1"
        assert "fusion_sweep" in ana_v3 and len(ana_v3["fusion_sweep"]) == \
            len(FUSION_SWEEP_WS)
        assert "popularity_train_delta" in ana_v3
        assert "bootstrap_cis_recall@100" in ana_v3 and \
            len(ana_v3["bootstrap_cis_recall@100"]) >= 1
        assert len(rows_v3) == len(c_uids)
        assert all("decay90_recall@100" in r for r in rows_v3)
        log(f"  v3/v4 evaluate: {len(res_v3)} scorers, "
            f"{len(ana_v3)} analyses, {len(rows_v3)} per-user rows PASSED")

        # ── 10. Verify v1_vocab.parquet was written ──
        assert v1_vocab_path.exists(), f"v1_vocab.parquet not found at {v1_vocab_path}"
        v1_check = pq.read_table(str(v1_vocab_path))
        assert "recording_mbid" in v1_check.column_names
        assert "count" in v1_check.column_names
        assert "v1_item_id" in v1_check.column_names
        assert len(v1_check) == n_v1

        # ── 11. v4.1 multi-seed + global-split smoke test ──
        # flatten + aggregate helpers
        flat = _flatten_primary(res_v3)
        assert "user_frequency" in flat and \
            "overall.recall@100" in flat["user_frequency"]
        agg = _mean_sd_across([flat["user_frequency"],
                               flat["user_frequency"],
                               flat["user_frequency"]])
        assert agg["overall.recall@100"]["sd"] == 0.0 and \
            agg["overall.recall@100"]["n"] == 3
        # global split on the tiny fixture: cutoff = max ts - 14d; the
        # fixture spans < 14 days of ts values, so cutoff may sit below
        # all events — every user then has post-cutoff items; assert the
        # mechanism, not the window semantics
        gs_test_v2, gs_cutoff = _global_split_test(
            events_path, c_uids, n_v2, cutoff_days=14)
        assert isinstance(gs_cutoff, int)
        assert len(gs_test_v2) >= 1, "global split found no post-cutoff users"
        # multiseed runner end-to-end on tiny fixtures (3 seeds)
        ms_out = tmpdir / "ms_out"
        ms_out.mkdir()
        run_multiseed_eval(listens_path, events_path, vocab_path,
                           emb_path, ms_out,
                           ranker_path=str(ranker_path),
                           bpr_path=str(bpr_path_t))
        ms = json.loads((ms_out / "multiseed.json").read_text())
        assert "42" in ms and "43" in ms and "44" in ms, "missing seed keys"
        assert "aggregate_mean_sd" in ms
        assert "user_frequency" in ms["aggregate_mean_sd"]
        gs = json.loads((ms_out / "globalsplit.json").read_text())
        assert "scorers" in gs and "user_frequency" in gs["scorers"]
        assert gs["cutoff_days"] == 14
        assert (ms_out / "metrics.json").exists(), "primary outputs missing"
        log("  v4.1 multiseed + globalsplit smoke PASSED")

        log(f"=== ALL ASSERTIONS PASSED (v1={n_v1}, v2={n_v2}) ===")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    if "--test" in sys.argv:
        _synthetic_test()
    else:
        main()
