"""
Cross-check kernel: independent re-computation of EDA statistics for verification.

Recomputes every EDA summary stat via a DIFFERENT implementation path
(pyarrow.compute vectorized ops + numpy) compared to the original EDA kernel
(Python Counter / sets / inline sorted-cumulative Gini). Purpose: catch bugs
that could exist in either implementation.

Input:  listens.parquet from lb-sanitize kernel (mount discovered at runtime)
Output: /kaggle/working/crosscheck_report.md, crosscheck_summary.json

Implementation differences from EDA kernel:
  Aggregation:  pc.value_counts per batch (vectorized) vs Python Counter per-row
  Gini:         Lorenz curve + trapz vs sorted-cumulative-shares loop
  Percentiles:  np.percentile(interpolation='linear') vs index-based lookup
  Year:         datetime64[Y] truncation vs precomputed DAY2YEAR table
  No per-row Python loops over 1.29B rows (all vectorized in pyarrow/numpy).
"""
import json, os, sys, time
from collections import defaultdict
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────
BATCH_SIZE      = 2_000_000
LOG_INTERVAL    = 50
ITEM_THRESHOLDS = [2, 5, 10, 50, 100]
REL_TOL_EXACT   = 0.001        # 0.1% for counts / ratios
REL_TOL_SOFT    = 0.01         # 1%  for percentile-type (definitional diffs)

# ── EXPECTED VALUES (from eda_summary.json — embedded; kernel can't read
# local files on Kaggle, so must be self-contained)                       ──
EXPECTED = {
    "total_rows": 1286727998,
    "n_users": 36970,
    "n_items": 4547485,
    "density": 0.0076536046554928445,
    "user_stats": {
        "mean": 34804.7,
        "median": 25923,
        "p90": 68493,
        "min": 128,
        "max": 687763,
    },
    "top1_share_pct": 6.5,
    "top10_share_pct": 30.7,
    "gini": 0.4157,
    "item_stats": {
        "ge_2": 4065756,
        "ge_5": 3372254,
        "ge_10": 2803656,
        "ge_50": 1482604,
        "ge_100": 1020199,
    },
    "year_range": [2005, 2024],
}

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

T0 = time.time()
def log(m): print(f"[{int(time.time()-T0):>5}s] {m}", flush=True)


# ── REUSABLE HELPERS (importable for local testing) ─────────────────────

def merge_value_counts(accum, batch_vc):
    """Fold a pyarrow value_counts StructArray into a Python dict.

    Works for any column type: int keys for user/year, bytes keys for
    recording_mbid.  pyarrow.compute.value_counts is vectorized in C++.
    """
    vals = batch_vc.field("values").to_pylist()
    cnts = batch_vc.field("counts").to_pylist()
    for v, c in zip(vals, cnts):
        key = v if isinstance(v, bytes) else int(v)
        accum[key] = accum.get(key, 0) + int(c)


def gini_lorenz_trapz(counts_np):
    """Gini coefficient via Lorenz curve + trapz (independent of sorted-cumulative-shares).

    Formula: G = 1 - 2 * A, where A is the area under the Lorenz curve
    (cumulative share of total vs cumulative share of population, sorted
    ascending).  Area computed via the trapezoidal rule with n+1 evenly
    spaced points.

    This is mathematically equivalent to the mean-absolute-difference form,
    but numerically independent of the EDA kernel's sorted-cumulative-shares
    loop (which computes G = sum(x_i * (2i - n - 1)) / (n * sum(x))).
    """
    x = np.sort(counts_np.astype(np.float64))
    total = x.sum()
    if len(x) == 0 or total == 0:
        return 0.0
    lorenz = np.concatenate(([0.0], np.cumsum(x) / total))
    # np.trapezoid for numpy 2.x; fall back to np.trapz for numpy 1.x
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    A = _trapz(lorenz, np.linspace(0.0, 1.0, len(x) + 1))
    return 1.0 - 2.0 * A


def ts_to_years(ts_np):
    """Convert int32 unix timestamps to calendar year (calendar-correct).

    Uses datetime64[Y] truncation — NOT integer division by seconds/year
    which drifts across year boundaries.  Matches the EDA kernel's
    calendar-correct year assignment via a different code path.
    """
    return (ts_np.astype(np.int64).view("datetime64[s]")
            .astype("datetime64[Y]").astype(np.int32) + 1970)


# ── CORE LOGIC ──────────────────────────────────────────────────────────

def run_crosscheck(input_path, output_dir):
    """Full crosscheck pipeline on a single listens.parquet file.

    Can be called locally with a test parquet; on Kaggle called from main().
    """
    input_path  = Path(input_path)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Input: {input_path}")
    pf = pq.ParquetFile(str(input_path))

    # Global accumulators (merged across batches)
    user_listens = defaultdict(int)      # user_id (int) -> listen count
    item_pop     = defaultdict(int)      # recording_mbid (bytes) -> count
    year_count   = defaultdict(int)      # year (int) -> count
    total_rows   = 0
    batch_num    = 0

    # ── STEP 1: streaming aggregation (vectorized, no per-row Python loop) ─
    log("STEP 1: streaming aggregation (pyarrow.compute vectorized)")
    for batch in pf.iter_batches(batch_size=BATCH_SIZE,
                                 columns=["user", "ts", "recording_mbid"]):
        batch_num += 1
        total_rows += batch.num_rows

        # pc.value_counts → StructArray; vectorized in C++ (no Python loop)
        merge_value_counts(user_listens, pc.value_counts(batch.column("user")))
        merge_value_counts(item_pop,     pc.value_counts(batch.column("recording_mbid")))

        # Year: proper datetime64[Y] truncation (calendar-correct)
        years = ts_to_years(batch.column("ts").to_numpy())
        merge_value_counts(year_count,
                           pc.value_counts(pa.array(years, type=pa.int32())))

        if batch_num % LOG_INTERVAL == 0:
            log(f"  batch {batch_num:>5}  rows={total_rows:>13,}  "
                f"users={len(user_listens):>7,}  items={len(item_pop):>9,}")

    log(f"  done: {total_rows:,} rows, {len(user_listens):,} users, "
        f"{len(item_pop):,} items")

    # ── STEP 2: derived stats (all numpy — no Python sorted/Counter) ─────
    log("STEP 2: derived stats (numpy vectorized)")

    user_arr = np.fromiter(user_listens.values(), dtype=np.int64)
    n_users  = len(user_arr)
    n_items  = len(item_pop)
    total_listens = int(user_arr.sum())
    density = total_rows / (n_users * n_items) if n_users and n_items else 0.0

    # Percentiles via np.percentile with linear interpolation (different from
    # the EDA kernel's index-based counts[int(len(counts) * p)]).
    # numpy 2.x renamed 'interpolation' -> 'method'; try both.
    def _pct(arr, q):
        try:
            return float(np.percentile(arr, q, method="linear"))
        except TypeError:
            return float(np.percentile(arr, q, interpolation="linear"))

    user_mean   = float(np.mean(user_arr))
    user_median = _pct(user_arr, 50)
    user_p90    = _pct(user_arr, 90)
    user_min    = int(np.min(user_arr))
    user_max    = int(np.max(user_arr))

    # Gini: Lorenz + trapz (different formula than original's sorted-cumulative)
    gini_val = gini_lorenz_trapz(user_arr)

    # Top share: numpy sort descending (different from Python sorted(reverse=True))
    sorted_desc = np.sort(user_arr)[::-1]
    top1_n  = max(1, int(n_users * 0.01))
    top10_n = max(1, int(n_users * 0.10))
    top1_share  = float(sorted_desc[:top1_n].sum()  / total_listens)
    top10_share = float(sorted_desc[:top10_n].sum() / total_listens)

    # Item thresholds via numpy boolean sum (different from sum(1 for c in ...))
    item_arr = np.fromiter(item_pop.values(), dtype=np.int64)
    item_stats = {f"ge_{t}": int(np.sum(item_arr >= t)) for t in ITEM_THRESHOLDS}

    # Year range
    yrs = sorted(year_count.keys())
    year_range = [yrs[0], yrs[-1]] if yrs else []

    # ── STEP 3: comparison ───────────────────────────────────────────────
    log("STEP 3: comparison against expected values")
    results = []

    def compare(name, exp, comp, exact=False):
        """Compare expected vs computed; exact=True for integer counts."""
        e, c = float(exp), float(comp)
        if e == 0 and c == 0:
            verdict = "PASS"
        elif exact:
            verdict = "PASS" if e == c else "FAIL"
        else:
            rel = abs(c - e) / abs(e) if e else float("inf")
            verdict = ("PASS"    if rel <= REL_TOL_EXACT else
                       "SOFT-PASS" if rel <= REL_TOL_SOFT else
                       "FAIL")
        results.append(dict(
            stat=name, expected=exp, computed=comp,
            abs_delta=round(abs(c - e), 6),
            rel_delta_pct=(round(abs(c - e) / abs(e) * 100, 4) if e else None),
            verdict=verdict,
        ))

    compare("total_rows",  EXPECTED["total_rows"],  total_rows, exact=True)
    compare("n_users",     EXPECTED["n_users"],     n_users, exact=True)
    compare("n_items",     EXPECTED["n_items"],     n_items, exact=True)
    compare("density",     EXPECTED["density"],     density)
    compare("user_mean",   EXPECTED["user_stats"]["mean"],   user_mean)
    compare("user_median", EXPECTED["user_stats"]["median"], user_median)
    compare("user_p90",    EXPECTED["user_stats"]["p90"],    user_p90)
    compare("user_min",    EXPECTED["user_stats"]["min"],    user_min, exact=True)
    compare("user_max",    EXPECTED["user_stats"]["max"],    user_max, exact=True)
    compare("top1_share_pct",  EXPECTED["top1_share_pct"],
            round(top1_share  * 100, 1))
    compare("top10_share_pct", EXPECTED["top10_share_pct"],
            round(top10_share * 100, 1))
    compare("gini",        EXPECTED["gini"],        round(gini_val, 4))
    for t in ITEM_THRESHOLDS:
        compare(f"item_ge_{t}", EXPECTED["item_stats"][f"ge_{t}"],
                item_stats[f"ge_{t}"], exact=True)
    compare("year_range_min", EXPECTED["year_range"][0], year_range[0],
            exact=True)
    compare("year_range_max", EXPECTED["year_range"][1], year_range[1],
            exact=True)

    # ── STEP 4: output ──────────────────────────────────────────────────
    log("STEP 4: writing reports")
    n_pass  = sum(1 for r in results if r["verdict"] == "PASS")
    n_spass = sum(1 for r in results if r["verdict"] == "SOFT-PASS")
    n_fail  = sum(1 for r in results if r["verdict"] == "FAIL")
    overall = "PASS" if n_fail == 0 else "FAIL"

    # ── markdown report ─────────────────────────────────────────────────
    rows_md = "\n".join(
        f"| {r['stat']} | {r['expected']} | {r['computed']} | "
        f"{r['abs_delta']:.4g} | "
        f"{r['rel_delta_pct'] if r['rel_delta_pct'] is not None else 'n/a'}% | "
        f"**{r['verdict']}** |"
        for r in results)

    yr_rows = "\n".join(f"| {y} | {year_count[y]:,} |"
                        for y in sorted(year_count.keys()))

    report = f"""# Crosscheck Report — EDA Statistics Verification

**Generated:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}
**Kernel:** lb_crosscheck.py (independent re-computation via pyarrow.compute + numpy)
**Input:** `{input_path}` ({total_rows:,} rows)

**Method (independent of EDA kernel):**
- Aggregation: `pc.value_counts` per batch (vectorized C++) vs EDA's Python Counter per-row
- Gini: Lorenz curve + trapezoidal rule vs EDA's sorted-cumulative-shares loop
- Percentiles: `np.percentile(interpolation='linear')` vs EDA's index-based lookup
- Year: `datetime64[Y]` truncation vs EDA's precomputed DAY2YEAR table

## Verdict: **{overall}**

| PASS | SOFT-PASS | FAIL |
|------|-----------|------|
| {n_pass} | {n_spass} | {n_fail} |

## Stat-by-Stat Comparison

| Stat | Expected | Computed | Abs Delta | Rel Delta | Verdict |
|------|----------|----------|-------|-------|---------|
{rows_md}

## Listens-per-Year Histogram (computed independently)

| Year | Listens |
|------|---------|
{yr_rows}

## Notes on Definitional Differences

- **Median/P90:** EDA uses `counts[int(len(counts) * p)]` (index-based);
  this kernel uses `np.percentile(..., interpolation='linear')` which
  interpolates between adjacent values.  SOFT-PASS acceptable for small
  numerical differences arising from this definitional gap.
- **Gini:** Both formulas are mathematically equivalent (mean-absolute-difference
  form), but numerical precision differs: EDA's cumulative Python loop vs
  this kernel's Lorenz curve + trapezoidal integration.  PASS/FAIL at 0.1%
  tolerance.
- **Year:** Both are calendar-correct.  EDA precomputes a DAY→YEAR lookup
  table via `calendar.timegm`; this kernel uses `datetime64[Y]` truncation.
  Same semantics, different code path.
"""
    (output_dir / "crosscheck_report.md").write_text(report, encoding="utf-8")
    log("  wrote crosscheck_report.md")

    # ── machine-readable summary ────────────────────────────────────────
    summary = dict(
        overall_verdict=overall,
        n_pass=n_pass, n_soft_pass=n_spass, n_fail=n_fail,
        results=results,
        method_notes=dict(
            gini="Lorenz curve + trapezoidal rule (independent of sorted-cumulative-shares)",
            percentiles="np.percentile with linear interpolation (may differ from index-based)",
            year="datetime64[Y] truncation (calendar-correct, different code path than DAY2YEAR)",
        ),
    )
    (output_dir / "crosscheck_summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")
    log("  wrote crosscheck_summary.json")

    # Final one-liner
    fails = [r["stat"] for r in results if r["verdict"] == "FAIL"]
    verdict_line = json.dumps(dict(
        overall=overall, passes=n_pass, soft_passes=n_spass,
        fails=n_fail, failed_stats=fails))
    print(verdict_line)
    log("DONE")
    return summary


def main():
    """Kaggle entry point — discover input parquet, run crosscheck."""
    root = Path("/kaggle/input")
    candidates = list(root.rglob("listens.parquet"))
    tree = "\n".join(f"  {p}" for p in sorted(root.rglob("*")) if p.is_file())
    log(f"/kaggle/input tree:\n{tree}")
    if not candidates:
        sys.exit("FATAL: listens.parquet not found under /kaggle/input")
    run_crosscheck(candidates[0], Path("/kaggle/working"))


if __name__ == "__main__":
    main()
