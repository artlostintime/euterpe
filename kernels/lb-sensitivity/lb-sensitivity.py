"""
Threshold-sensitivity kernel: count flagged users in the sanitized listens
at multiple flood/repeat thresholds, without changing the published pipeline.

Reads data/clean/listens.parquet from lb-sanitize output (kernel source).
For each user x (day, recording) and (hour, recording) cell we need counts;
vectorized via pandas-free pyarrow + numpy groupby on hashed keys.

v2 (round-3 review fixes):
  - Cross-batch cell flush: the parquet is user-sorted and batches are 2M rows,
    so a user spans at most 2 consecutive batches. Cells split across a batch
    boundary previously failed the per-batch floor filter in BOTH batches.
    Now per-user cell counts accumulate in a `pending` dict and are floor-filtered
    + merged into the global Counters only when the user is complete (a new
    batch's minimum user id proves all lower users are finished).
  - Distinct-recordings-per-user-hour (the PUBLISHED flood criterion) is now
    computed alongside the same-recording intensity table.

Outputs: reports/sensitivity_report.md + sensitivity_summary.json
Run `python lb_sensitivity.py --test` locally for the synthetic check.
"""
import collections
import json, sys, time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ── CONFIG ─────────────────────────────────────────────────────
DAY_THRESHOLDS   = [10, 25, 50, 100, 250]
HOUR_THRESHOLDS  = [100, 250, 500, 1000]      # same-recording intensity
DISTINCT_HOUR_THRESHOLDS = [100, 250, 500, 1000]  # published flood criterion
SECONDS_DAY      = 86400
SECONDS_HOUR     = 3600
BATCH_ROWS       = 2_000_000

DAY_MIN  = min(DAY_THRESHOLDS)      # 10
HOUR_MIN = min(HOUR_THRESHOLDS)     # 100
HOUR_BASE = 300_000                 # hour buckets start ~307871 (2005) — safe lower bound

T0 = time.time()
def log(m): print(f"[{int(time.time()-T0):>5}s] {m}", flush=True)

# ── core: per-user pending flush merge ─────────────────────────
# pending_day[user] = {packed_day_key: count}   (raw counts, no floor yet)
# pending_hour[user] = {packed_hour_key: count}
# A user is complete when a later batch proves no more rows for them exist
# (user-sorted file: min user id of the NEXT batch > user).

day_cells  = collections.Counter()   # packed day key -> count (cells >= DAY_MIN)
hour_cells = collections.Counter()   # packed hour key -> count (cells >= HOUR_MIN)
# distinct recordings per (user, hour): derived from hour cells >= 1 — but we
# only keep hour cells >= HOUR_MIN for the intensity table. For the distinct
# criterion we need ALL distinct recs per user-hour, so track separately:
# distinct_hour[(user, hour)] = set of rec hashes — bounded: only user-hours
# with >= 100 distinct recs can matter (min threshold), and a user-hour holds
# at most ~3600 events, so sets are tiny. We keep per-user pending distinct
# sets and flush complete users, keeping only user-hours with >= min threshold.
DISTINCT_MIN = min(DISTINCT_HOUR_THRESHOLDS)  # 100

def flush_user(u, pday, phour, pdist):
    """Floor-filter one complete user's pending cells into the global Counters."""
    for k, c in pday.pop(u, {}).items():
        if c >= DAY_MIN:
            day_cells[k] += c
    for k, c in phour.pop(u, {}).items():
        if c >= HOUR_MIN:
            hour_cells[k] += c
    for k, c in pdist.pop(u, {}).items():
        if c >= DISTINCT_MIN:
            distinct_hour[k] = c

def flush_complete(min_new_user, pday, phour, pdist):
    """Flush every pending user strictly below min_new_user (they are complete)."""
    for u in [x for x in pday if x < min_new_user]:
        flush_user(u, pday, phour, pdist)

def process_batch(u, ts, rh32, pday, phour, pdist):
    """Accumulate one batch's cell counts into the per-user pending dicts."""
    day = ts // SECONDS_DAY
    hour = ts // SECONDS_HOUR
    if (hour < HOUR_BASE).any():
        sys.exit(f"FATAL: hour bucket {int(hour.min())} below HOUR_BASE {HOUR_BASE}")

    # packed 1-D int64 keys (day: 15 bits + rh32 32 bits = 47 -> user at bit 47)
    key_d = (u << 47) | (day << 32) | rh32
    du, dc = np.unique(key_d, return_counts=True)
    for k, c in zip(du.tolist(), dc.tolist()):
        uid = int(k) >> 47
        pd = pday[uid]
        pd[k] = pd.get(k, 0) + c

    rh24 = rh32 >> 8                          # top 3 bytes
    key_h = (u << 42) | ((hour - HOUR_BASE) << 24) | rh24
    hu, hc = np.unique(key_h, return_counts=True)
    for k, c in zip(hu.tolist(), hc.tolist()):
        uid = int(k) >> 42
        ph = phour[uid]
        ph[k] = ph.get(k, 0) + c

    # distinct recordings per (user, hour) — FULL 32-bit rec hash (not rh24:
    # the 24-bit cell hash would collapse recordings sharing top-3 bytes).
    # Exact via lexsort: count distinct (user, hour, rh32) triples per (user, hour).
    hb = hour - HOUR_BASE
    order = np.lexsort((rh32, hb, u))
    su_, sh_, sr_ = u[order], hb[order], rh32[order]
    new_triple = np.empty(len(su_), dtype=bool)
    new_triple[0] = True
    if len(su_) > 1:
        np.logical_or(su_[1:] != su_[:-1], sh_[1:] != sh_[:-1], out=new_triple[1:])
        np.logical_or(new_triple[1:], sr_[1:] != sr_[:-1], out=new_triple[1:])
    new_uh = np.empty(len(su_), dtype=bool)
    new_uh[0] = True
    if len(su_) > 1:
        np.logical_or(su_[1:] != su_[:-1], sh_[1:] != sh_[:-1], out=new_uh[1:])
    idx = np.flatnonzero(new_uh)
    # distinct recs per (user, hour) group = sum of new_triple within the group
    counts = np.add.reduceat(new_triple.astype(np.int64), idx)
    # packed (user, hour) key: user at bit 18, hour-bucket in low 18 bits
    uhk = (su_[idx].astype(np.int64) << 18) | sh_[idx]
    for k, c in zip(uhk.tolist(), counts.tolist()):
        uid = int(k) >> 18
        pu = pdist[uid]
        pu[k] = pu.get(k, 0) + c

def run(input_path):
    pending_day, pending_hour, pending_dist = (collections.defaultdict(dict),
                                               collections.defaultdict(dict),
                                               collections.defaultdict(dict))
    global distinct_hour
    distinct_hour = {}   # (user, hour_bucket) -> distinct-rec count (>= DISTINCT_MIN)

    pf = pq.ParquetFile(input_path)
    n = 0
    for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=["user", "ts", "recording_mbid"]):
        u = batch.column("user").to_numpy(zero_copy_only=False).astype(np.int64)
        ts = batch.column("ts").to_numpy(zero_copy_only=False).astype(np.int64)
        rec_arr = batch.column("recording_mbid")
        parts = []
        for c in rec_arr.chunks if isinstance(rec_arr, pa.ChunkedArray) else [rec_arr]:
            parts.append(np.frombuffer(c.buffers()[1], dtype=np.uint8, count=len(c) * 16,
                                       offset=c.offset * 16).reshape(-1, 16)[:, :4].copy().view(np.uint32).reshape(-1))
        rh32 = np.concatenate(parts) if len(parts) > 1 else parts[0]
        rh32 = rh32.astype(np.int64)

        if n > 0:
            flush_complete(int(u.min()), pending_day, pending_hour, pending_dist)
        process_batch(u, ts, rh32, pending_day, pending_hour, pending_dist)

        n += len(u)
        if (n // BATCH_ROWS) % 50 == 0:
            log(f"  rows {n:,} day_cells {len(day_cells):,} hour_cells {len(hour_cells):,} "
                f"distinct_hours {len(distinct_hour):,} pending_users {len(pending_day):,}")

    # end of stream: every remaining pending user is complete
    for uid in list(pending_day.keys()):
        flush_user(uid, pending_day, pending_hour, pending_dist)

    log(f"pass done: {n:,} rows; day_cells {len(day_cells):,}; hour_cells {len(hour_cells):,}; "
        f"distinct user-hours {len(distinct_hour):,}")

    # flagged users per threshold (user id = top bits of packed key)
    def flagged(counter, thr, shift):
        users = set()
        for k, c in counter.items():
            if c > thr:
                users.add(int(k) >> shift)
        return len(users)

    def flagged_distinct(thr):
        users = set()
        for k, c in distinct_hour.items():
            if c > thr:
                users.add(int(k) >> 18)
        return len(users)

    day_results  = {t: flagged(day_cells, t, 47) for t in DAY_THRESHOLDS}
    hour_results = {t: flagged(hour_cells, t, 42) for t in HOUR_THRESHOLDS}
    distinct_results = {t: flagged_distinct(t) for t in DISTINCT_HOUR_THRESHOLDS}
    return n, day_results, hour_results, distinct_results


def write_outputs(n, day_results, hour_results, distinct_results, out_dir):
    out = Path(out_dir)
    rep = out / "reports"; rep.mkdir(parents=True, exist_ok=True)
    summary = dict(
        rows=n,
        day_thresholds=DAY_THRESHOLDS, day_flagged_users=day_results,
        hour_thresholds=HOUR_THRESHOLDS, hour_flagged_users=hour_results,
        distinct_hour_thresholds=DISTINCT_HOUR_THRESHOLDS,
        distinct_hour_flagged_users=distinct_results,
        note="day cells: per (user, calendar-day, recording) event counts (exact match to repeat-day flag at >50). "
             "hour table: same-recording events per user-hour (intensity criterion, NOT the published flag). "
             "distinct-hour table: distinct recordings per user-hour — the PUBLISHED flood criterion (>500). "
             "Cells split across parquet batch boundaries are combined via per-user pending flush before "
             "floor filtering (v2 fix). Recording keys truncated to 4 bytes (collisions only possible within "
             "a single (user, bucket) cell — negligible; could only merge two tracks' counts, slight overcount).",
    )
    (rep / "sensitivity_report.md").write_text(
        "# Noise-flag threshold sensitivity\n\n"
        "## Repeat-day (events of same recording per user-day) > T\n\n"
        "| T | flagged users |\n|---|---|\n"
        + "".join(f"| {t} | {day_results[t]:,} |\n" for t in DAY_THRESHOLDS)
        + "\n## Same-recording-per-hour (intensity criterion, not the published flag) > T\n\n"
        "| T | flagged users |\n|---|---|\n"
        + "".join(f"| {t} | {hour_results[t]:,} |\n" for t in HOUR_THRESHOLDS)
        + "\n## Distinct-recordings-per-hour (PUBLISHED flood criterion) > T\n\n"
        "| T | flagged users |\n|---|---|\n"
        + "".join(f"| {t} | {distinct_results[t]:,} |\n" for t in DISTINCT_HOUR_THRESHOLDS)
        + "\n" + summary["note"] + "\n", encoding="utf-8")
    (out / "sensitivity_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log("DONE")
    print(json.dumps(summary))
    return summary


def main():
    root = Path("/kaggle/input")
    cands = [p for p in root.rglob("listens.parquet")]
    if not cands: sys.exit("FATAL: listens.parquet not found under /kaggle/input")
    if len(cands) > 1: sys.exit(f"FATAL: multiple listens.parquet found: {cands}")
    log(f"input: {cands[0]}")
    n, d, h, dist = run(cands[0])
    write_outputs(n, d, h, dist, "/kaggle/working")


# ── synthetic test ──────────────────────────────────────────────
def _synthetic_test():
    import tempfile, os
    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(42)

    # users 0..3; user 2 has a day-cell SPLIT across two batches (6 + 6 events)
    # user 3 has a user-hour with 120 distinct recordings (flood criterion)
    rows = []
    def add(u, ts, rec):
        rows.append((u, ts, rec))

    # user 0: normal, 12 events same rec same day (one batch only)
    for i in range(12): add(0, 1_600_000_000 + i * 60, b"\x01" * 16)
    # user 1: below floor everywhere
    for i in range(5):  add(1, 1_600_000_000 + i * 60, b"\x02" * 16)
    # user 2: 6 events at day X, 6 events at day X (same day! same rec) — split across batches
    day_ts = 1_600_000_000 // 86400 * 86400
    for i in range(6):  add(2, day_ts + i * 60, b"\x03" * 16)
    for i in range(6):  add(2, day_ts + i * 60 + 3600, b"\x03" * 16)  # same calendar day
    # user 3: 120 distinct recs within one hour (distinct criterion > 100)
    hour_ts = 1_600_000_000 // 3600 * 3600
    for i in range(120): add(3, hour_ts + i, bytes([i % 256, 1, 2, 3]) + b"\x00" * 12)

    users = np.array([r[0] for r in rows], dtype=np.int32)
    tss   = np.array([r[1] for r in rows], dtype=np.int32)
    recs  = np.array([r[2] for r in rows], dtype=object)
    tbl = pa.table(dict(user=users, ts=tss,
                        recording_mbid=pa.array([bytes(r) for r in recs], type=pa.binary(16))))
    pq.write_table(tbl, tmp / "listens.parquet")

    # run with a tiny batch size so user 2's rows land in 2 batches
    global BATCH_ROWS
    BATCH_ROWS = 10
    n, d, h, dist = run(tmp / "listens.parquet")

    # brute force — key definitions MUST mirror the kernel's documented hashes:
    #   day cells: full 4-byte hash (bytes 0-3)
    #   hour intensity cells: 24-bit hash (bytes 1-3; bit-packing constraint)
    #   distinct recordings: full 4-byte hash
    from collections import Counter as Ctr
    day_true = Ctr(); hour_true = Ctr(); dist_true = {}
    for u_, ts_, rec_ in rows:
        day_true[(u_, ts_ // 86400, rec_[:4])] += 1
        hour_true[(u_, ts_ // 3600, rec_[1:4])] += 1
    per_uh = {}
    for (u_, hb, rec_) in hour_true:
        pass  # distinct needs the FULL hash — recompute from rows below
    for u_, ts_, rec_ in rows:
        per_uh.setdefault((u_, ts_ // 3600), set()).add(rec_[:4])
    for k, v in per_uh.items():
        if len(v) >= DISTINCT_MIN:
            dist_true[k] = len(v)

    def bf_flag_day(t):
        return len({k[0] for k, c in day_true.items() if c > t})
    def bf_flag_hour(t):
        return len({k[0] for k, c in hour_true.items() if c > t})
    def bf_flag_dist(t):
        return len({k[0] for k, c in dist_true.items() if c > t})

    assert n == len(rows), f"row count {n} != {len(rows)}"
    # cross-batch fix: user 2's split cell (12 events) must survive floor 10
    assert any(k[0] == 2 and c >= 12 for k, c in day_true.items()), "brute-force sanity"
    assert d[10] == bf_flag_day(10), f"day@10 {d[10]} != {bf_flag_day(10)}"
    assert d[25] == bf_flag_day(25), f"day@25 {d[25]} != {bf_flag_day(25)}"
    assert h[100] == bf_flag_hour(100), f"hour@100 {h[100]} != {bf_flag_hour(100)}"
    assert dist[100] == bf_flag_dist(100), f"dist@100 {dist[100]} != {bf_flag_dist(100)}"
    assert dist[500] == bf_flag_dist(500), f"dist@500 {dist[500]} != {bf_flag_dist(500)}"
    # user 3 must be flagged at distinct>100 but not >500
    assert bf_flag_dist(100) >= 1 and bf_flag_dist(500) == 0
    # user 2 flagged at day>10 (12 events) — the cross-batch case
    assert bf_flag_day(10) >= 1

    # write outputs to tmp and verify files exist
    write_outputs(n, d, h, dist, tmp)
    assert (tmp / "reports" / "sensitivity_report.md").exists()
    assert (tmp / "sensitivity_summary.json").exists()
    print("=== ALL SENSITIVITY TESTS PASSED ===")
    import shutil; shutil.rmtree(tmp)


if __name__ == "__main__":
    if "--test" in sys.argv:
        _synthetic_test()
    else:
        main()
