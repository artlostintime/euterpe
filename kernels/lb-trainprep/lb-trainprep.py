"""
Training-prep kernel: build item vocabulary + per-user temporal splits.

Reads sanitized MLHD+ listens.parquet (~1.29B rows) and produces:
  - data/prep/events.parquet  (user, ts, item_id, split) — filtered to vocab items
  - data/prep/split_summary.json
  - reports/trainprep_report.md
  - README.md

Pipeline (all vectorized pyarrow/numpy, batch_size 2M, no per-row Python loops):
  Pass 1: global item counts + per-user max_ts
  Pass 2: filter to vocab (count >= 10), map recording_mbid -> item_id,
           assign temporal splits, stream-write output parquet.

Split definitions (per user, relative to that user's max_ts):
  train = rel >= 28d   (item_id maps to 0)
  val   = 14d <= rel < 28d   (1)
  test  = rel < 14d   (2)

Input: listens.parquet from lb-sanitize (discovered at runtime via rglob).
Output: /kaggle/working/data/prep/*, /kaggle/working/reports/*, README.md
"""
import json, os, sys, time
from collections import defaultdict
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────
BATCH_SIZE   = 2_000_000
LOG_INTERVAL = 50
MIN_COUNT    = 10               # item vocabulary threshold
VAL_DAYS     = 14
TRAIN_DAYS   = 28
SECONDS_DAY  = 86400

# ── EXPECTED VALUES (embedded; kernel can't read local files on Kaggle) ─
EXPECTED = dict(
    total_rows=1286727998,
    n_users=36970,
    n_items=4547485,
    vocab_size=2803656,         # items with count >= 10
    oov_upper_bound_pct=1.22,   # 1.22% = ~15.7M rows
)

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

T0 = time.time()
def log(m): print(f"[{int(time.time()-T0):>5}s] {m}", flush=True)


# ── REUSABLE HELPERS ────────────────────────────────────────────────────

def merge_value_counts(accum, batch_vc):
    """Fold a pyarrow value_counts StructArray into a Python dict.

    Works for any column type: int keys for user, bytes keys for
    recording_mbid.  pyarrow.compute.value_counts is vectorized in C++.
    """
    vals = batch_vc.field("values").to_pylist()
    cnts = batch_vc.field("counts").to_pylist()
    for v, c in zip(vals, cnts):
        key = v if isinstance(v, bytes) else int(v)
        accum[key] = accum.get(key, 0) + int(c)


def merge_user_max_ts(accum, batch_table):
    """Vectorized group-by-user max(ts) via pc.group_by, merged into dict.

    Uses pyarrow's group_by kernel (vectorized C++), not a Python loop.
    """
    if batch_table.num_rows == 0:
        return
    # RecordBatch -> Table (cheap; RecordBatch lacks group_by)
    if not hasattr(batch_table, "group_by"):
        batch_table = pa.table({col: batch_table.column(col)
                                for col in batch_table.schema.names})
    agg = batch_table.group_by("user").aggregate([("ts", "max")])
    users = agg.column("user").to_pylist()
    maxes = agg.column("ts_max").to_pylist()
    for u, m in zip(users, maxes):
        u = int(u)
        accum[u] = m if u not in accum else max(accum[u], m)


def build_vocab(item_counts, min_count=MIN_COUNT):
    """Build vocabulary: items with count >= min_count.

    Returns:
      vocab_mbid_to_id: dict[bytes, int32] — recording_mbid -> item_id
      vocab_table: pa.Table with columns (recording_mbid, count, item_id),
                   sorted by (-count, recording_mbid) for determinism.
    """
    eligible = [(mbid, cnt) for mbid, cnt in item_counts.items()
                if cnt >= min_count]
    # Sort: primary key = -count (descending), secondary = mbid (ascending bytes)
    eligible.sort(key=lambda x: (-x[1], x[0]))
    mbid_to_id = {mbid: i for i, (mbid, _) in enumerate(eligible)}
    vocab_table = pa.table({
        "recording_mbid": pa.array([e[0] for e in eligible], type=pa.binary(16)),
        "count": pa.array([e[1] for e in eligible], type=pa.int64()),
        "item_id": pa.array(list(range(len(eligible))), type=pa.int32()),
    })
    return mbid_to_id, vocab_table


def assign_splits(ts_np, max_ts):
    """Vectorized split assignment for one batch of events.

    Args:
      ts_np: numpy int32 array of timestamps
      max_ts: int (single user) or int64 array aligned with ts_np (per-user max)
    Returns:
      numpy int8 array: 0=train, 1=val, 2=test
    """
    rel = np.asarray(max_ts, dtype=np.int64) - ts_np.astype(np.int64)
    split = np.full(len(ts_np), 0, dtype=np.int8)  # default train
    # Order matters: test (smallest window) first, then val overwrites where needed
    split[rel < VAL_DAYS * SECONDS_DAY]   = 2        # test: < 14d
    mask_val = (rel >= VAL_DAYS * SECONDS_DAY) & (rel < TRAIN_DAYS * SECONDS_DAY)
    split[mask_val] = 1                               # val: 14d <= rel < 28d
    # Remaining (rel >= 28d) stay 0 = train
    return split


# ── CORE PIPELINE ───────────────────────────────────────────────────────

def run_trainprep(input_path, output_dir):
    """Full training-prep pipeline.  Called from main() on Kaggle or locally."""
    input_path  = Path(input_path)
    output_dir  = Path(output_dir)
    prep_dir    = output_dir / "data" / "prep"
    rep_dir     = output_dir / "reports"
    prep_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    log(f"Input: {input_path}")

    # ════════════════════════════════════════════════════════════════════
    # PASS 1 — global aggregation (item counts + per-user max_ts)
    # ════════════════════════════════════════════════════════════════════
    log("PASS 1: global aggregation")
    pf = pq.ParquetFile(str(input_path))
    item_counts  = defaultdict(int)  # recording_mbid (bytes) -> count
    user_max_ts  = {}                # user (int) -> max timestamp
    total_rows   = 0
    batch_num    = 0

    for batch in pf.iter_batches(batch_size=BATCH_SIZE,
                                 columns=["user", "ts", "recording_mbid"]):
        batch_num += 1
        total_rows += batch.num_rows

        merge_value_counts(item_counts,
                           pc.value_counts(batch.column("recording_mbid")))
        merge_user_max_ts(user_max_ts, batch)

        if batch_num % LOG_INTERVAL == 0:
            log(f"  batch {batch_num:>5}  rows={total_rows:>13,}  "
                f"items={len(item_counts):>9,}  users={len(user_max_ts):>7,}")

    log(f"  pass 1 done: {total_rows:,} rows, {len(user_max_ts):,} users, "
        f"{len(item_counts):,} distinct items")

    # Sanity check
    actual_rows = total_rows
    actual_users = len(user_max_ts)
    actual_items = len(item_counts)
    log(f"  expected rows={EXPECTED['total_rows']:,}  actual={actual_rows:,}  "
        f"{'OK' if actual_rows == EXPECTED['total_rows'] else 'MISMATCH'}")
    log(f"  expected users={EXPECTED['n_users']:,}  actual={actual_users:,}  "
        f"{'OK' if actual_users == EXPECTED['n_users'] else 'MISMATCH'}")
    log(f"  expected items={EXPECTED['n_items']:,}  actual={actual_items:,}  "
        f"{'OK' if actual_items == EXPECTED['n_items'] else 'MISMATCH'}")

    # Free item_counts memory — only needed for the sanity check above
    del item_counts

    # Per-user max_ts lookup table (user ids are small sequential ints).
    # Replaces per-row dict lookups with O(1) numpy fancy indexing.
    max_ts_lut = np.zeros(max(user_max_ts.keys()) + 1, dtype=np.int64)
    for u, m in user_max_ts.items():
        max_ts_lut[u] = m

    # ════════════════════════════════════════════════════════════════════
    # TRAIN-COUNT PASS — item counts from TRAIN split only (no leakage)
    # ════════════════════════════════════════════════════════════════════
    # Vocab eligibility must not see holdout activity: pass 1 counts all
    # events (for integrity checks only); the vocabulary is built from
    # train-split events exclusively. Train = rel >= TRAIN_DAYS window.
    log("TRAIN-COUNT PASS: item counts from train split only")
    item_counts_train = defaultdict(int)
    train_events = 0
    pf2 = pq.ParquetFile(str(input_path))
    train_cut = TRAIN_DAYS * SECONDS_DAY
    for batch in pf2.iter_batches(batch_size=BATCH_SIZE,
                                  columns=["user", "ts", "recording_mbid"]):
        users_np = np.asarray(batch.column("user"), dtype=np.int64)
        ts_np    = np.asarray(batch.column("ts"), dtype=np.int64)
        rel = max_ts_lut[users_np] - ts_np
        train_mask = rel >= train_cut
        train_events += int(train_mask.sum())
        if train_mask.any():
            rec_col = batch.column("recording_mbid")
            rec_train = rec_col.take(np.nonzero(train_mask)[0])
            merge_value_counts(item_counts_train,
                               pc.value_counts(rec_train))
    log(f"  train-count pass done: {train_events:,} train events, "
        f"{len(item_counts_train):,} distinct train items")

    # ════════════════════════════════════════════════════════════════════
    # VOCABULARY — train items with count >= MIN_COUNT
    # ════════════════════════════════════════════════════════════════════
    log(f"Building vocabulary (min_count={MIN_COUNT}, train-only)")
    mbid_to_id, vocab_table = build_vocab(item_counts_train, MIN_COUNT)
    vocab_size = len(mbid_to_id)
    log(f"  vocab size: {vocab_size:,}  "
        f"(previous all-events vocab: {EXPECTED['vocab_size']:,}, "
        f"delta {abs(vocab_size - EXPECTED['vocab_size']):,})")

    # Write vocab for downstream kernels
    vocab_path = prep_dir / "vocab.parquet"
    pq.write_table(vocab_table, str(vocab_path))
    log(f"  wrote {vocab_path}")

    del item_counts_train

    # ════════════════════════════════════════════════════════════════════
    # PASS 2 — filter, map, split, stream-write events.parquet
    # ════════════════════════════════════════════════════════════════════
    log("PASS 2: filter + split + write")
    pf = pq.ParquetFile(str(input_path))  # reopen for second pass

    out_schema = pa.schema([
        ("user",    pa.int32()),
        ("ts",      pa.int32()),
        ("item_id", pa.int32()),
        ("split",   pa.int8()),
    ])
    writer = pq.ParquetWriter(
        str(prep_dir / "events.parquet"), out_schema,
        compression="zstd", use_dictionary=False)

    # Statistics accumulators
    events_in      = 0
    events_out     = 0
    oov_dropped    = 0
    oov_per_split  = defaultdict(int)
    split_counts   = defaultdict(int)   # split_id -> event count
    split_users    = defaultdict(set)   # split_id -> set of users

    # Order-check state
    last_user        = -1
    user_regressions = 0

    # Sample-based ordering check (100 random users)
    import random
    rng = random.Random(42)
    # Pick 100 users from user_max_ts keys (deterministic)
    all_users = sorted(user_max_ts.keys())
    sample_users = set(rng.sample(all_users, min(100, len(all_users))))
    sample_ts_state = {}   # user -> last_ts seen (for ordering check)
    sample_violations = 0

    batch_num = 0
    for batch in pf.iter_batches(batch_size=BATCH_SIZE,
                                 columns=["user", "ts", "recording_mbid"]):
        batch_num += 1
        events_in += batch.num_rows

        users_np  = batch.column("user").to_numpy()
        ts_np     = batch.column("ts").to_numpy()
        recs_list = batch.column("recording_mbid").to_pylist()  # binary -> list of bytes

        # Map recording_mbid -> item_id via dict lookup.
        # For 2M rows per batch, a list comprehension is ~0.3s — acceptable.
        item_ids  = [mbid_to_id.get(r, -1) for r in recs_list]
        item_ids_np = np.array(item_ids, dtype=np.int32)
        del recs_list  # free memory

        # OOV mask
        oov_mask = item_ids_np == -1
        n_oov = int(oov_mask.sum())
        oov_dropped += n_oov

        # Filter to in-vocab rows
        in_vocab = ~oov_mask
        if not in_vocab.any():
            continue

        users_f = users_np[in_vocab]
        ts_f    = ts_np[in_vocab]
        items_f = item_ids_np[in_vocab]

        # Assign splits (vectorized per batch, via max_ts LUT fancy indexing)
        splits_f = assign_splits(ts_f, max_ts_lut[users_f])

        # Track split statistics
        for s_id in range(3):
            mask = splits_f == s_id
            n = int(mask.sum())
            split_counts[s_id] += n
            if n > 0:
                split_users[s_id].update(np.unique(users_f[mask]).tolist())

        # Order-check: user regressions across batches
        batch_users_unique = np.unique(users_np)
        batch_first_user = int(batch_users_unique[0]) if len(batch_users_unique) > 0 else -1
        if batch_first_user < last_user and last_user >= 0:
            user_regressions += 1
        last_user = int(batch_users_unique[-1]) if len(batch_users_unique) > 0 else last_user

        # Sample-based ts ordering check
        sample_mask = np.isin(users_f, list(sample_users))
        if sample_mask.any():
            su = users_f[sample_mask]
            st = ts_f[sample_mask]
            for i in range(len(su)):
                u = int(su[i])
                t = int(st[i])
                if u in sample_ts_state:
                    if t < sample_ts_state[u]:
                        sample_violations += 1
                sample_ts_state[u] = t

        # Write filtered batch
        out_batch = pa.table({
            "user":    pa.array(users_f, type=pa.int32()),
            "ts":      pa.array(ts_f, type=pa.int32()),
            "item_id": pa.array(items_f, type=pa.int32()),
            "split":   pa.array(splits_f, type=pa.int8()),
        })
        writer.write_table(out_batch)
        events_out += len(users_f)

        # OOV per split (approximate: based on what split WOULD have been assigned)
        if n_oov > 0:
            oov_splits = assign_splits(ts_np[oov_mask],
                                       max_ts_lut[users_np[oov_mask]])
            for s_id in range(3):
                oov_per_split[s_id] += int((oov_splits == s_id).sum())

        if batch_num % LOG_INTERVAL == 0:
            log(f"  batch {batch_num:>5}  in={events_in:>13,}  "
                f"out={events_out:>13,}  oov={oov_dropped:>11,}")

    writer.close()
    log(f"  pass 2 done: {events_in:,} in, {events_out:,} out, "
        f"{oov_dropped:,} OOV dropped ({oov_dropped/events_in*100:.2f}%)")

    # ════════════════════════════════════════════════════════════════════
    # PASS 3 — verification + summary
    # ════════════════════════════════════════════════════════════════════
    log("PASS 3: verification")

    # Split boundary check on sample users (re-read from written parquet)
    log("  checking split boundaries on sample users...")
    boundary_ok = True
    boundary_violations = 0
    # Read back only sample users via filters
    pf_out = pq.ParquetFile(str(prep_dir / "events.parquet"))
    user_split_data = defaultdict(lambda: defaultdict(list))  # user -> split -> [ts]
    sample_arr = np.array(sorted(sample_users), dtype=np.int32)
    for batch in pf_out.iter_batches(batch_size=BATCH_SIZE,
                                      columns=["user", "ts", "split"]):
        users_b = batch.column("user").to_numpy()
        mask = np.isin(users_b, sample_arr)
        if not mask.any():
            continue
        ts_b  = batch.column("ts").to_numpy()[mask]
        spl_b = batch.column("split").to_numpy().astype(np.int8)[mask]
        for u, t, s in zip(users_b[mask].tolist(), ts_b.tolist(),
                           spl_b.tolist()):
            user_split_data[u][int(s)].append(int(t))

    for u in sample_users:
        sd = user_split_data.get(u, {})
        train_ts = sd.get(0, [])
        val_ts   = sd.get(1, [])
        test_ts  = sd.get(2, [])
        if train_ts and val_ts:
            if max(train_ts) > min(val_ts):
                boundary_violations += 1
                boundary_ok = False
        if val_ts and test_ts:
            if max(val_ts) > min(test_ts):
                boundary_violations += 1
                boundary_ok = False
    log(f"  split boundary check: {len(sample_users)} users, "
        f"{boundary_violations} violations  {'OK' if boundary_ok else 'FAIL'}")

    # Users with empty train / thin train
    # (Need a third pass or track during pass 2 — we tracked per-split user sets)
    users_all = set(user_max_ts.keys())
    users_train = split_users.get(0, set())
    users_val   = split_users.get(1, set())
    users_test  = split_users.get(2, set())
    no_train_users = len(users_all - users_train)

    # For thin-train count, we need per-user train event counts.
    # Quick third pass over just user+split columns, counting train events per user.
    log("  counting per-user train events (light pass)...")
    user_train_counts = defaultdict(int)
    pf_out2 = pq.ParquetFile(str(prep_dir / "events.parquet"))
    for batch in pf_out2.iter_batches(batch_size=BATCH_SIZE,
                                       columns=["user", "split"]):
        users_b = batch.column("user").to_numpy()
        spl_b   = batch.column("split").to_numpy()
        train_mask = spl_b == 0
        if train_mask.any():
            tu = users_b[train_mask]
            # Vectorized count per user
            unique_users, counts = np.unique(tu, return_counts=True)
            for u, c in zip(unique_users.tolist(), counts.tolist()):
                user_train_counts[u] += c
    thin_train_users = sum(1 for c in user_train_counts.values() if c < 10)
    log(f"  users with empty train: {no_train_users:,}")
    log(f"  users with < 10 train events: {thin_train_users:,}")

    # ════════════════════════════════════════════════════════════════════
    # OUTPUT — reports + summary
    # ════════════════════════════════════════════════════════════════════
    log("WRITING reports")

    summary = dict(
        vocab_size=vocab_size,
        expected_vocab_size=EXPECTED["vocab_size"],
        vocab_match=abs(vocab_size - EXPECTED["vocab_size"]) / EXPECTED["vocab_size"] < 0.01,
        total_rows_in=events_in,
        total_rows_out=events_out,
        oov_dropped=oov_dropped,
        oov_pct=round(oov_dropped / events_in * 100, 4) if events_in else 0,
        oov_per_split={str(k): v for k, v in oov_per_split.items()},
        split_counts={str(k): v for k, v in split_counts.items()},
        split_users={str(k): len(v) for k, v in split_users.items()},
        n_users_total=len(users_all),
        users_no_train=no_train_users,
        users_thin_train_lt10=thin_train_users,
        order_check=dict(
            user_regressions=user_regressions,
            ts_violations_sampled=sample_violations,
            sample_size=len(sample_users),
        ),
        boundary_check=dict(
            ok=boundary_ok,
            violations=boundary_violations,
            sample_size=len(sample_users),
        ),
    )

    (prep_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")
    log("  wrote split_summary.json")

    # Human-readable report
    split_labels = {0: "train", 1: "val", 2: "test"}
    split_table = "\n".join(
        f"| {split_labels[s]} | {split_counts[s]:,} | "
        f"{len(split_users[s]):,} |"
        for s in [0, 1, 2])

    report = f"""# Training-Prep Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}
**Kernel:** lb_trainprep.py
**Input:** `{input_path}` ({events_in:,} rows)

## Overview

| Metric | Value |
|--------|-------|
| Input rows | {events_in:,} |
| Output rows | {events_out:,} |
| OOV dropped | {oov_dropped:,} ({oov_dropped/events_in*100:.2f}%) |
| Vocabulary size | {vocab_size:,} (min_count={MIN_COUNT}) |
| Distinct users | {len(users_all):,} |

## Vocabulary

| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| Vocab size | {EXPECTED['vocab_size']:,} | {vocab_size:,} | {"PASS" if summary['vocab_match'] else "FAIL"} |

## Split Distribution

| Split | Events | Users |
|-------|--------|-------|
{split_table}

## User Coverage

| Metric | Count |
|--------|-------|
| Users with no train events | {no_train_users:,} |
| Users with < 10 train events | {thin_train_users:,} |

## Order Checks

| Check | Result |
|-------|--------|
| User-ID regressions across batches | {user_regressions} {"OK" if user_regressions == 0 else "FAIL"} |
| TS ordering violations (100-user sample) | {sample_violations} {"OK" if sample_violations == 0 else "FAIL"} |
| Split boundary violations (100-user sample) | {boundary_violations} {"OK" if boundary_ok else "FAIL"} |

## Split Definitions

- **test**: rel < {VAL_DAYS} days before user's max ts (most recent events)
- **val**: {VAL_DAYS}d <= rel < {TRAIN_DAYS}d before user's max ts
- **train**: rel >= {TRAIN_DAYS}d before user's max ts (oldest events)

## Output Files

| File | Description |
|------|-------------|
| `data/prep/vocab.parquet` | Vocabulary: recording_mbid, count, item_id |
| `data/prep/events.parquet` | Filtered events: user, ts, item_id, split |
| `data/prep/split_summary.json` | Machine-readable statistics |
"""
    (rep_dir / "trainprep_report.md").write_text(report, encoding="utf-8")
    log("  wrote trainprep_report.md")

    # README
    (output_dir / "README.md").write_text(f"""# Music Recommender — Training Prep Kernel

Builds item vocabulary (count >= {MIN_COUNT}) and per-user temporal splits
from sanitized MLHD+ listening history.

## Usage by Training Kernels

```python
import pyarrow.parquet as pq
vocab = pq.read_table("data/prep/vocab.parquet")   # recording_mbid, count, item_id
events = pq.read_table("data/prep/events.parquet")  # user, ts, item_id, split
train = events.filter(events["split"] == 0)
```

**Split semantics:** test = last {VAL_DAYS} days per user, val = days {VAL_DAYS}-{TRAIN_DAYS},
train = everything older. Deterministic via vocabulary sort order.

**Rerun:** Kaggle CPU kernel, no internet. All outputs deterministic.
""", encoding="utf-8")
    log("  wrote README.md")

    # Final one-liner
    final = dict(
        vocab_size=vocab_size,
        events_in=events_in,
        events_out=events_out,
        oov_dropped=oov_dropped,
        split_counts={str(k): v for k, v in split_counts.items()},
    )
    print(json.dumps(final))
    log("DONE")
    return summary


def main():
    """Kaggle entry point — discover input parquet, run training prep."""
    root = Path("/kaggle/input")
    candidates = list(root.rglob("listens.parquet"))
    tree = "\n".join(f"  {p}" for p in sorted(root.rglob("*")) if p.is_file())
    log(f"/kaggle/input tree:\n{tree}")
    if not candidates:
        sys.exit("FATAL: listens.parquet not found under /kaggle/input")
    run_trainprep(candidates[0], Path("/kaggle/working"))


if __name__ == "__main__":
    main()
