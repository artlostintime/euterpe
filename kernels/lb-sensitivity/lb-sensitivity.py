"""
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Shuvi

Threshold-sensitivity kernel: count flagged users in the sanitized listens
at multiple flood/repeat thresholds, without changing the published pipeline.

Reads data/clean/listens.parquet from lb-sanitize output (kernel source).
For each user x (day, recording) and (hour, recording) cell we need counts;
vectorized via pandas-free pyarrow + numpy groupby on hashed keys.

Outputs: reports/sensitivity_report.md + sensitivity_summary.json
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ── CONFIG ─────────────────────────────────────────────────────
DAY_THRESHOLDS   = [10, 25, 50, 100, 250]
HOUR_THRESHOLDS  = [100, 250, 500, 1000]
SECONDS_DAY      = 86400
SECONDS_HOUR     = 3600
BATCH_ROWS       = 2_000_000

T0 = time.time()
def log(m): print(f"[{int(time.time()-T0):>5}s] {m}", flush=True)

# ── discovery (assert exactly one) ────────────────────────────
root = Path("/kaggle/input")
cands = [p for p in root.rglob("listens.parquet")]
if not cands: sys.exit("FATAL: listens.parquet not found under /kaggle/input")
if len(cands) > 1: sys.exit(f"FATAL: multiple listens.parquet found: {cands}")
INPUT = cands[0]
log(f"input: {INPUT}")

# ── vectorized pass: per (user, day-bucket, recording) and (user, hour-bucket, recording) counts ──
# Strategy: accumulate a global dict of cell -> count is too big. Instead:
# per-user-day-recording counts via sorted-array groupby per batch is wrong across batches.
# Correct + lazy: two passes over the parquet using np.unique on full concatenated keys is memory-heavy.
# Chosen: per-batch compute (user, bucket, rec) 16-byte structured keys, accumulate counts in a
# dict of numpy arrays via chunk merge (np.unique per batch + add into global dict keyed by bytes).
# Given 1.29B rows, dict-of-bytes->int is ~hundreds of millions entries worst case -> too big.
# PRACTICAL ALTERNATIVE (exact, bounded): only need FLAGGED-USER COUNTS per threshold.
# A user is flagged at day threshold T iff any (user, day, rec) count > T.
# Equivalent: count per (user, day, rec) pairs. We stream batches, keep a per-batch unique,
# and merge into a global Counter ONLY of cells whose count >= min(thresholds) seen so far
# (sparse: only heavy cells survive). Two-phase per batch: unique counts, then filter to
# >= min(DAY_THRESHOLDS) before merging into the global dict. Global dict stays small
# (only cells >= 10/day). Hourly analog with >= 100.
import collections

DAY_MIN  = min(DAY_THRESHOLDS)      # 10
HOUR_MIN = min(HOUR_THRESHOLDS)     # 100
HOUR_BASE = 300_000                 # hour buckets start ~307871 (2005) — safe lower bound

day_cells  = collections.Counter()   # packed day key -> count (only cells >= 10)
hour_cells = collections.Counter()   # packed hour key -> count (only cells >= 100)

def merge(counter, keys, counts, floor):
    # only survivors (count >= floor) ever become Python objects — keeps this tiny
    mask = counts >= floor
    for k, c in zip(keys[mask].tolist(), counts[mask].tolist()):
        counter[k] += c

pf = pq.ParquetFile(INPUT)
n = 0
for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=["user", "ts", "recording_mbid"]):
    u = batch.column("user").to_numpy(zero_copy_only=False).astype(np.int64)
    ts = batch.column("ts").to_numpy(zero_copy_only=False).astype(np.int64)
    rec_arr = batch.column("recording_mbid")
    # first 4 bytes of recording mbid as 32-bit key (collisions only matter within
    # a (user, bucket) cell — negligible at a few hundred distinct recs per cell)
    parts = []
    for c in rec_arr.chunks if isinstance(rec_arr, pa.ChunkedArray) else [rec_arr]:
        parts.append(np.frombuffer(c.buffers()[1], dtype=np.uint8, count=len(c) * 16,
                                   offset=c.offset * 16).reshape(-1, 16)[:, :4].copy().view(np.uint32).reshape(-1))
    rh32 = np.concatenate(parts) if len(parts) > 1 else parts[0]
    rh32 = rh32.astype(np.int64)                     # < 2^32, safe in int64

    day = ts // SECONDS_DAY
    hour = ts // SECONDS_HOUR
    if (hour < HOUR_BASE).any():
        sys.exit(f"FATAL: hour bucket {int(hour.min())} below HOUR_BASE {HOUR_BASE}")

    # packed 1-D int64 keys -> fast sort-based np.unique (vs slow axis=0 structured unique)
    # day needs 15 bits (max 19955) + rh32 32 bits = 47 -> user at bit 47 (no overlap!)
    key_d = (u << 47) | (day << 32) | rh32                       # 63 bits
    du, dc = np.unique(key_d, return_counts=True)
    merge(day_cells, du, dc, DAY_MIN)

    rh24 = rh32 >> 8                                             # top 3 bytes
    key_h = (u << 42) | ((hour - HOUR_BASE) << 24) | rh24        # 59 bits
    hu, hc = np.unique(key_h, return_counts=True)
    merge(hour_cells, hu, hc, HOUR_MIN)

    n += len(u)
    if (n // BATCH_ROWS) % 50 == 0:
        log(f"  rows {n:,} day_cells {len(day_cells):,} hour_cells {len(hour_cells):,}")

log(f"pass done: {n:,} rows; day_cells {len(day_cells):,}; hour_cells {len(hour_cells):,}")

# flagged users per threshold (user id = top bits of packed key)
def flagged(counter, thr, shift):
    users = set()
    for k, c in counter.items():
        if c > thr:
            users.add(int(k) >> shift)
    return len(users)

day_results  = {t: flagged(day_cells, t, 47) for t in DAY_THRESHOLDS}
hour_results = {t: flagged(hour_cells, t, 42) for t in HOUR_THRESHOLDS}

out = Path("/kaggle/working")
rep = out / "reports"; rep.mkdir(parents=True, exist_ok=True)
summary = dict(
    rows=n, users=None,
    day_thresholds=DAY_THRESHOLDS, day_flagged_users=day_results,
    hour_thresholds=HOUR_THRESHOLDS, hour_flagged_users=hour_results,
    note="day cells: per (user, calendar-day, recording) counts of events (exact match to repeat-day flag at >50); "
         "hour cells: per (user, hour, recording) counts of the SAME recording, NOT distinct recordings — "
         "the published flood flag counts distinct recordings per hour; this table measures same-recording intensity per hour instead. "
         "Recording keys truncated to 4 bytes (collisions only possible within a single (user, bucket) cell — negligible; could only merge two tracks' counts, slight overcount).",
)
(rep / "sensitivity_report.md").write_text(
    "# Noise-flag threshold sensitivity\n\n"
    "## Repeat-day (events of same recording per user-day) > T\n\n"
    "| T | flagged users |\n|---|---|\n"
    + "".join(f"| {t} | {day_results[t]:,} |\n" for t in DAY_THRESHOLDS)
    + "\n## Same-recording-per-hour > T\n\n"
    "| T | flagged users |\n|---|---|\n"
    + "".join(f"| {t} | {hour_results[t]:,} |\n" for t in HOUR_THRESHOLDS)
    + "\n" + summary["note"] + "\n", encoding="utf-8")
(out / "sensitivity_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
log("DONE")
print(json.dumps(summary))
