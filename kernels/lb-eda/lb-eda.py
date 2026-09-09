"""
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Shuvi

Phase 3 kernel (Kaggle): exploratory data analysis on sanitized MLHD+ listens.

Reads lb-sanitize kernel output (listens.parquet, ~1.287B rows).
Streaming aggregations via pyarrow ParquetFile.iter_batches.
Figures saved to figures/.

Inputs:
  /kaggle/input/lb-sanitize/data/clean/listens.parquet  (schema: user, ts, recording_mbid, release_mbid)
  /kaggle/input/lb-sanitize/data/clean/users.parquet    (user_int -> user_uuid)

Outputs (/kaggle/working):
  reports/eda_report.md
  figures/long_tail.png, lorenz.png, listens_year.png
  eda_summary.json
  README.md
"""
import json, os, sys, time, calendar
from pathlib import Path
from collections import Counter
from itertools import accumulate

# ── CONFIG ──────────────────────────────────────────────────────────────
SAMPLE_USERS = 1000
SEED        = 42
BATCH_SIZE  = 2_000_000
ITEM_THRESHOLDS = [2, 5, 10, 50, 100]   # recording listen-count thresholds

WORK  = Path("/kaggle/working")
FIG   = WORK / "figures"
REP   = WORK / "reports"
for d in (FIG, REP):
    d.mkdir(parents=True, exist_ok=True)

import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T0 = time.time()
def log(m): print(f"[{int(time.time()-T0):>5}s] {m}", flush=True)
def gb(n): return f"{n/1e9:.2f} GB" if n >= 1e9 else f"{n/1e6:.1f} MB"

# Input discovered at runtime (mount name of kernel data source varies)
root = Path("/kaggle/input")
candidates = [p for p in root.rglob("listens.parquet")]
tree = "\n".join(f"  {p}" for p in sorted(root.rglob("*")) if p.is_file())
log(f"/kaggle/input tree:\n{tree}")
if not candidates:
    sys.exit("FATAL: no listens.parquet found under /kaggle/input")
if len(candidates) > 1:
    sys.exit(f"FATAL: multiple listens.parquet found, expected exactly one: {candidates}")
INPUT_PQ = candidates[0]


def gini(values):
    """Gini coefficient from a list of non-negative values. O(n log n)."""
    if not values or sum(values) == 0:
        return 0.0
    s = sorted(values)
    n = len(s)
    total = sum(s)
    cum = 0.0
    for i, v in enumerate(s):
        cum += v * (2 * (i + 1) - n - 1)
    return cum / (n * total)


# ════════════════════════════════════════════════════════════════════════
# STEP 1: streaming aggregation pass
# ════════════════════════════════════════════════════════════════════════
log("STEP 1: streaming aggregation pass")
pf = pq.ParquetFile(str(INPUT_PQ))

# per-user counters
user_listens   = Counter()    # user -> listen count
user_first_ts  = {}           # user -> min ts
user_last_ts   = {}           # user -> max ts

# global item popularity
item_pop = Counter()          # recording_mbid (bytes) -> count

# temporal
year_count = Counter()        # year -> count
# exact UTC day -> year table (no per-row gmtime; integer-div is fast)
_YEARS = list(range(2004, 2028))
_YSTARTS = [calendar.timegm((y, 1, 1, 0, 0, 0)) // 86400 for y in _YEARS]
DAY2YEAR = {}
for _i, _y in enumerate(_YEARS):
    _end = _YSTARTS[_i + 1] if _i + 1 < len(_YSTARTS) else _YSTARTS[-1] + 366
    for _d in range(_YSTARTS[_i], _end):
        DAY2YEAR[_d] = _y

total_rows = 0
batch_num  = 0

# sampled per-user stats
import random
rng = random.Random(SEED)
sample_users_set = set()       # filled on first batch from encountered user IDs
sample_distinct = Counter()    # user -> set size (tracked as count of unique recs)
sample_repeats  = Counter()    # (user, recording) -> repeat count (only for sampled users)
# For sampled users, track distinct recordings as a set per user
sample_user_recs = {}          # user -> set of recording_mbid bytes

for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["user", "ts", "recording_mbid"]):
    batch_num += 1
    users  = batch.column("user").to_pylist()
    ts_arr = batch.column("ts").to_pylist()
    recs   = batch.column("recording_mbid").to_pylist()

    for i in range(len(users)):
        u = users[i]
        t = ts_arr[i]
        r = recs[i]       # bytes, 16 bytes, always present

        total_rows += 1

        # per-user
        user_listens[u] += 1
        if u not in user_first_ts or t < user_first_ts[u]:
            user_first_ts[u] = t
        if u not in user_last_ts or t > user_last_ts[u]:
            user_last_ts[u] = t

        # item popularity
        item_pop[r] += 1

        # temporal (exact UTC year via day table)
        year_count[DAY2YEAR.get(t // 86400, 2004 if t < 1108339204 else 2026)] += 1

        # sampled per-user distinct/repeat tracking
        # lazy-init sample set from first SAMPLE_USERS distinct users seen
        if len(sample_users_set) < SAMPLE_USERS:
            if u not in sample_users_set:
                sample_users_set.add(u)
                sample_user_recs[u] = set()
        if u in sample_users_set:
            rec_set = sample_user_recs[u]
            if r in rec_set:
                sample_repeats[u] += 1
            else:
                rec_set.add(r)

    if batch_num % 50 == 0:
        log(f"  batch {batch_num}, {total_rows:,} rows, {len(user_listens):,} users, {len(item_pop):,} items")

log(f"  total: {total_rows:,} rows, {len(user_listens):,} users, {len(item_pop):,} distinct recordings")


# ════════════════════════════════════════════════════════════════════════
# STEP 2: derived stats
# ════════════════════════════════════════════════════════════════════════
log("STEP 2: derived stats")
counts = sorted(user_listens.values())
def pct(p):
    return counts[min(len(counts) - 1, int(len(counts) * p))] if counts else 0

user_stats = {
    "mean": round(sum(counts) / len(counts), 1) if counts else 0,
    "median": pct(0.5), "p90": pct(0.9),
    "min": min(counts) if counts else 0,
    "max": max(counts) if counts else 0,
}

# sparsity
n_users = len(user_listens)
n_items = len(item_pop)
density = total_rows / (n_users * n_items) if (n_users and n_items) else 0

# power-user bias: top-1%, top-10% share
sorted_counts = sorted(user_listens.values(), reverse=True)
total_listens = sum(sorted_counts)
top1_n  = max(1, int(n_users * 0.01))
top10_n = max(1, int(n_users * 0.10))
top1_share  = sum(sorted_counts[:top1_n]) / total_listens if total_listens else 0
top10_share = sum(sorted_counts[:top10_n]) / total_listens if total_listens else 0
gini_val = gini(counts)

# lorenz curve (50 evenly spaced cumulative-share points, ascending order)
sorted_counts_asc = sorted(user_listens.values())
cum_sums = list(accumulate(sorted_counts_asc))
lorenz_x = [i / 50 for i in range(51)]
lorenz_y = [0.0] + [cum_sums[min(int(i * n_users / 50), n_users - 1)] / total_listens
                     if total_listens else 0.0 for i in range(1, 51)]

# item popularity stats
pop_counts = sorted(item_pop.values(), reverse=True)
item_stats = {}
for thr in ITEM_THRESHOLDS:
    item_stats[f"ge_{thr}"] = sum(1 for c in pop_counts if c >= thr)

# sampled per-user stats
sample_distinct_list = [len(sample_user_recs[u]) for u in sample_users_set if u in sample_user_recs]
sample_listen_list   = [user_listens[u] for u in sample_users_set if u in user_listens]
sample_ratio_list    = [sample_listen_list[i] / sample_distinct_list[i]
                        if sample_distinct_list[i] > 0 else 0
                        for i in range(len(sample_distinct_list))]
sample_repeat_list   = [sample_repeats[u] for u in sample_users_set if u in sample_repeats]

sampled_stats = {
    "n_users": len(sample_users_set),
    "mean_distinct": round(sum(sample_distinct_list) / len(sample_distinct_list), 1) if sample_distinct_list else 0,
    "mean_listen_distinct_ratio": round(sum(sample_ratio_list) / len(sample_ratio_list), 2) if sample_ratio_list else 0,
    "repeat_median": sorted(sample_repeat_list)[len(sample_repeat_list) // 2] if sample_repeat_list else 0,
    "repeat_p90": sorted(sample_repeat_list)[int(len(sample_repeat_list) * 0.9)] if sample_repeat_list else 0,
    "repeat_max": max(sample_repeat_list) if sample_repeat_list else 0,
}


# ════════════════════════════════════════════════════════════════════════
# STEP 3: figures
# ════════════════════════════════════════════════════════════════════════
log("STEP 3: figures")

# --- long_tail.png ---
fig, ax = plt.subplots(figsize=(8, 5))
rank_vals = list(range(1, len(pop_counts) + 1))
ax.loglog(rank_vals, pop_counts, linewidth=0.6, alpha=0.8)
ax.set_xlabel("Item rank (log)")
ax.set_ylabel("Listen count (log)")
ax.set_title("Long-tail distribution of recording popularity")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "long_tail.png", dpi=150)
plt.close(fig)
log("  saved long_tail.png")

# --- lorenz.png ---
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(lorenz_x, lorenz_y, linewidth=1.5, label="Lorenz curve")
ax.plot([0, 1], [0, 1], linewidth=1, linestyle="--", color="gray", label="Perfect equality")
ax.fill_between(lorenz_x, lorenz_y, lorenz_x, alpha=0.15)
ax.set_xlabel("Cumulative share of users (sorted by listens, ascending)")
ax.set_ylabel("Cumulative share of total listens")
ax.set_title(f"Lorenz curve — Gini = {gini_val:.3f}")
ax.legend(loc="upper left")
ax.annotate(f"Top 1% = {top1_share*100:.1f}% of listens", xy=(0.99, top1_share),
            xytext=(0.75, top1_share + 0.05), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black"))
ax.annotate(f"Top 10% = {top10_share*100:.1f}% of listens", xy=(0.90, top10_share),
            xytext=(0.60, top10_share + 0.05), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black"))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "lorenz.png", dpi=150)
plt.close(fig)
log("  saved lorenz.png")

# --- listens_year.png ---
fig, ax = plt.subplots(figsize=(10, 5))
years_sorted = sorted(year_count.keys())
yr_vals = [year_count[y] for y in years_sorted]
ax.bar(years_sorted, yr_vals, color="#4C72B0", edgecolor="white", linewidth=0.3)
ax.set_xlabel("Year")
ax.set_ylabel("Number of listens")
ax.set_title("Listens per year")
ax.tick_params(axis="x", rotation=45)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "listens_year.png", dpi=150)
plt.close(fig)
log("  saved listens_year.png")


# ════════════════════════════════════════════════════════════════════════
# STEP 4: reports + summary
# ════════════════════════════════════════════════════════════════════════
log("STEP 4: reports")

year_rows = "".join(f"| {y} | {year_count[y]:,} |\n" for y in sorted(year_count.keys()))
thresh_rows = "".join(f"| ≥{thr} | {item_stats[f'ge_{thr}']:,} ({item_stats[f'ge_{thr}']/n_items*100:.1f}%) |\n"
                      for thr in ITEM_THRESHOLDS)

(REP / "eda_report.md").write_text(f"""# EDA Report — MLHD+ Complete Shard

**Generated:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}
**Kernel:** lb_eda.py (Phase 3 — EDA)
**Input:** `{INPUT_PQ}` ({total_rows:,} rows)

## Overview

| Metric | Value |
|---|---|
| Total listens | {total_rows:,} |
| Distinct users | {n_users:,} |
| Distinct recordings | {n_items:,} |
| Sparsity (density) | {density:.2e} |
| Matrix size (users × items) | {n_users:,} × {n_items:,} = {n_users * n_items:.2e} |

## Per-user Listen Count Distribution

| Stat | Value |
|---|---|
| Mean | {user_stats['mean']} |
| Median | {user_stats['median']:,} |
| P90 | {user_stats['p90']:,} |
| Min | {user_stats['min']:,} |
| Max | {user_stats['max']:,} |

## Power-user Bias

| Metric | Value |
|---|---|
| Top 1% of users ({top1_n:,}) listen share | {top1_share*100:.1f}% |
| Top 10% of users ({top10_n:,}) listen share | {top10_share*100:.1f}% |
| Gini coefficient | {gini_val:.3f} |

## Listens per Year

| Year | Listens |
|---|---|
{year_rows}
## Item Popularity (recording listen counts)

| Threshold | Recordings ≥ threshold | % of all items |
|---|---|---|
{thresh_rows}
## Sampled Per-user Distinct/Repeat Stats ({len(sample_users_set)} users)

Diagnostic subsample: the first {SAMPLE_USERS} distinct users in file order
(parquet is user-sorted), NOT a random sample. Figures are indicative, not estimates.

| Stat | Value |
|---|---|
| Mean distinct recordings per user | {sampled_stats['mean_distinct']:,} |
| Mean listens/distinct ratio | {sampled_stats['mean_listen_distinct_ratio']} |
| Repeat same-recording (median) | {sampled_stats['repeat_median']:,} |
| Repeat same-recording (P90) | {sampled_stats['repeat_p90']:,} |
| Repeat same-recording (max) | {sampled_stats['repeat_max']:,} |

## Known Limitations

1. **Sampled per-user stats (diagnostic subsample):** distinct/recording-repeat stats computed on the first {SAMPLE_USERS} distinct users in file order (parquet is user-sorted), not a random sample (memory safety: full-user analysis would require ~O(users × distinct_items) memory).
2. **Single shard:** MLHD+ complete-f only; other shards may differ in user distribution and item coverage.
3. **Year derivation:** year computed via an exact UTC day-to-year boundary table (calendar.timegm), matching calendar years exactly; UTC only, not timezone-aware.
""")

summary = {
    "total_rows": total_rows, "n_users": n_users, "n_items": n_items,
    "density": density,
    "user_stats": user_stats,
    "top1_share_pct": round(top1_share * 100, 1),
    "top10_share_pct": round(top10_share * 100, 1),
    "gini": round(gini_val, 4),
    "item_stats": {f"ge_{t}": item_stats[f"ge_{t}"] for t in ITEM_THRESHOLDS},
    "sampled_stats": sampled_stats,
    "year_range": [min(year_count.keys()), max(year_count.keys())] if year_count else [],
    "figures": ["figures/long_tail.png", "figures/lorenz.png", "figures/listens_year.png"],
}
(WORK / "eda_summary.json").write_text(json.dumps(summary, indent=1))

(WORK / "README.md").write_text(f"""# Music Recommender — Phase 3 (EDA Kernel)

Exploratory data analysis on sanitized MLHD+ complete-shard listens.
Reads lb-sanitize kernel output; produces report, figures, and machine-readable summary.

**Input:** `{INPUT_PQ}` (from lb-sanitize kernel)
**Output:** `reports/eda_report.md`, `figures/*.png`, `eda_summary.json`

Rerun: Kaggle kernel, CPU, no internet needed. All figures deterministic.
""")

log("DONE — all outputs written")
print(json.dumps({k: summary[k] for k in
      ("total_rows", "n_users", "n_items", "density",
       "top1_share_pct", "top10_share_pct", "gini")}, indent=1))
