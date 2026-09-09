"""
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Shuvi

Phase 2 kernel (Kaggle): sanitize ListenBrainz MLHD+ COMPLETE listening-history shard.

Downloads mlhdplus-complete-f.tar, validates timestamps and IDs, deduplicates,
detects noise users, writes clean parquet output + quality reports.

Column semantics (musicbrainz.org/doc/MLHD%2B):
  col0 = unix timestamp (seconds)
  col1 = artist_mbids (comma-separated UUIDs when multiple, may be empty)
  col2 = release_mbid (uuid or empty)
  col3 = recording_mbid (uuid or empty — expected ~100% in complete shard)

No MSIDs in this dataset. No mapping join needed.

Spec: sanitization pipeline — single file, Kaggle CPU script kernel.

Outputs (/kaggle/working):
  data/clean/listens.parquet           clean, deduplicated listening events
  data/clean/users.parquet             user_int → user_uuid mapping
  reports/data_quality_report.md       full quality report
  summary.json                         machine-readable stats
  README.md                            brief description
"""
import json, os, sys, time, hashlib, tarfile, shutil, platform, urllib.request, re, subprocess, uuid
from pathlib import Path
from collections import Counter, defaultdict

# ── CONFIG ──────────────────────────────────────────────────────────────
MLHD_URL = "https://data.metabrainz.org/pub/musicbrainz/listenbrainz/mlhd/mlhdplus-complete-f.tar"
MLHD_MD5 = "0f50c57561c2adf651492b1230da459d"
MLHD_BYTES = 15903508480                # expected file size for size-check
TS_MIN = 946684800                      # 2000-01-01 00:00:00 UTC
FUTURE_SLOP = 86400                     # 1 day
NOISE_HOURLY = 500                      # distinct recordings/hour threshold
NOISE_DAILY_REPEAT = 50                 # same-recording/day threshold
BATCH_ROWS = 2_000_000                  # parquet write batch size

WORK = Path("/kaggle/working")
SCRATCH = Path("/kaggle/tmp")           # scratch space for the 15.9GB tar (NOT under working quota)
CLEAN = WORK / "data" / "clean"
REP = WORK / "reports"
for d in (SCRATCH, CLEAN, REP):
    d.mkdir(parents=True, exist_ok=True)

try:
    import zstandard as zstd
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zstandard"], check=True)
    import zstandard as zstd

import pyarrow as pa
import pyarrow.parquet as pq

T0 = time.time()
def log(m): print(f"[{int(time.time()-T0):>5}s] {m}", flush=True)
def gb(n): return f"{n/1e9:.2f} GB" if n >= 1e9 else f"{n/1e6:.1f} MB"

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
# artist_mbids: comma-separated UUIDs, no trailing comma, allow empty
ARTIST_MBIDS_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"(,[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})*$", re.I)


def uuid_to_bytes(s):
    """Convert UUID string to 16 bytes, or None if empty/invalid."""
    if not s:
        return None
    try:
        return uuid.UUID(s).bytes
    except ValueError:
        return None


def download_and_verify(url, expected_md5, dest, label):
    """Stream-download + MD5 verify. Reuse existing file if size matches."""
    h = hashlib.md5()
    if dest.exists() and dest.stat().st_size == MLHD_BYTES:
        log(f"  {label}: file present ({gb(dest.stat().st_size)}); hashing only")
        with open(dest, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 23), b""):
                h.update(chunk)
    else:
        log(f"  {label}: downloading ...")
        with urllib.request.urlopen(url, timeout=600) as r, open(dest, "wb") as f:
            done = 0
            while True:
                chunk = r.read(1 << 23)
                if not chunk:
                    break
                f.write(chunk); h.update(chunk); done += len(chunk)
                if done // (200 << 20) != (done - len(chunk)) // (200 << 20):
                    log(f"  {label}: {gb(done)}")
        log(f"  {label}: download done ({gb(done)})")
    got = h.hexdigest()
    log(f"  {label}: md5 expected={expected_md5}  got={got}")
    if got != expected_md5:
        sys.exit(f"CHECKSUM MISMATCH — {label}")
    return dest


# ════════════════════════════════════════════════════════════════════════
# STEP A: download
# ════════════════════════════════════════════════════════════════════════
log("STEP A: download + md5 verify")
tar_path = download_and_verify(MLHD_URL, MLHD_MD5,
                               SCRATCH / "mlhdplus-complete-f.tar", "MLHD+ complete")


# ════════════════════════════════════════════════════════════════════════
# STEP B: per-user streaming pass
# ════════════════════════════════════════════════════════════════════════
log("STEP B: per-user streaming pass")
# Frozen audit cutoff (2026-09-01 00:00 UTC, fixed before execution) — keeps the
# future-timestamp acceptance window deterministic across re-runs. The 3
# future-dated events in the complete shard (2029/2031/2035) exceed any
# plausible cutoff and remain rejected regardless.
NOW = 1788220800 + FUTURE_SLOP
dctx = zstd.ZstdDecompressor()

tf = tarfile.open(str(tar_path), "r:")
user_members = sorted(
    [m for m in tf.getmembers() if m.name.endswith(".txt.zst")],
    key=lambda m: m.name
)
log(f"  {len(user_members):,} user files (sorted)")

# --- counters ---
raw_listens = 0
rejected = Counter()
malformed_artist_count = 0      # rows where artist col was non-empty but didn't match pattern (stat only; artist not stored)
malformed_release_count = 0     # kept rows where release was non-empty but not valid uuid
duplicates_removed = 0
clean_count = 0
users_total = 0
users_retained = 0
users_dropped = 0
ts_clean_min = None; ts_clean_max = None
recording_mbid_count = 0        # retained rows with valid recording_mbid

# noise tracking
flood_flagged = {}
repeat_flagged = {}

# user mapping + listen counts
user_rows = []
user_listen_counts = Counter()

# rejected-line examples (first 3 per reason, for the report)
rej_examples = defaultdict(list)
def reject(reason, line):
    rejected[reason] += 1
    if len(rej_examples[reason]) < 3:
        rej_examples[reason].append(line)

# parquet writer — streaming, ~2M rows per batch
listen_schema = pa.schema([
    ("user", pa.int32()),
    ("ts", pa.int32()),              # unix-sec, int32 safe until 2038
    ("recording_mbid", pa.binary(16)),  # always present (validated)
    ("release_mbid", pa.binary(16)),    # null when absent
])
listen_pq = CLEAN / "listens.parquet"
listen_writer = pq.ParquetWriter(str(listen_pq), listen_schema, compression="zstd")
buf_user = []; buf_ts = []; buf_rec = []; buf_rel = []


def flush_listens():
    global buf_user, buf_ts, buf_rec, buf_rel, buf_art
    if not buf_user:
        return
    tbl = pa.table({
        "user":          pa.array(buf_user, pa.int32()),
        "ts":            pa.array(buf_ts, pa.int32()),
        "recording_mbid": pa.array(buf_rec, pa.binary(16)),
        "release_mbid":  pa.array(buf_rel, pa.binary(16)),
    })
    listen_writer.write_table(tbl)
    buf_user.clear(); buf_ts.clear(); buf_rec.clear(); buf_rel.clear()


for user_int, m in enumerate(user_members):
    user_uuid = os.path.basename(m.name).replace(".txt.zst", "")
    user_rows.append((user_int, user_uuid))
    users_total += 1

    with tf.extractfile(m) as fh:
        txt = dctx.stream_reader(fh).read().decode("utf-8", "replace")

    seen = set()
    hour_distinct = defaultdict(set)
    day_rec_count = defaultdict(Counter)
    has_clean = False

    for line in txt.splitlines():
        if not line.strip():
            continue
        raw_listens += 1
        cols = line.split("\t")

        # accept 2/3/4+ cols, pad missing with empty
        if len(cols) < 2:
            reject("bad_col_count", line)
            continue
        while len(cols) < 4:
            cols.append("")

        # timestamp (col0)
        try:
            ts = int(cols[0].strip())
        except (ValueError, IndexError):
            reject("bad_ts", line)
            continue
        if not (TS_MIN <= ts <= NOW):
            reject("ts_out_of_range", line)
            continue

        # recording_mbid (col3) — must be valid uuid in complete shard
        recording_raw = cols[3].strip()
        if recording_raw and not UUID_RE.match(recording_raw):
            reject("malformed_recording", line)
            continue
        recording = recording_raw or None

        # artist_mbids (col1) — not stored (derivable from MB tables via recording_mbid);
        # parsed only to count malformed values as a data-quality stat
        artist_raw = cols[1].strip()
        if artist_raw and not ARTIST_MBIDS_RE.match(artist_raw):
            malformed_artist_count += 1

        # release_mbid (col2) — valid uuid or empty
        release_raw = cols[2].strip()
        release = None
        if release_raw:
            if UUID_RE.match(release_raw):
                release = release_raw
            else:
                malformed_release_count += 1

        # complete shard: recording MBID guaranteed by construction — reject if absent
        if recording is None:
            reject("no_recording_mbid", line)
            continue

        # dedup key: (timestamp, recording_mbid)
        dedup_key = (ts, recording)
        if dedup_key in seen:
            duplicates_removed += 1
            continue
        seen.add(dedup_key)

        # recording coverage tracking (retained rows only)
        recording_mbid_count += 1

        # noise tracking
        hour_key = ts // 3600
        day_key = ts // 86400
        hour_distinct[hour_key].add(recording)
        day_rec_count[day_key][recording] += 1

        # emit clean row
        has_clean = True
        clean_count += 1
        user_listen_counts[user_int] += 1
        buf_user.append(user_int); buf_ts.append(ts)
        buf_rec.append(uuid_to_bytes(recording))
        buf_rel.append(uuid_to_bytes(release))

        if ts_clean_min is None or ts < ts_clean_min:
            ts_clean_min = ts
        if ts_clean_max is None or ts > ts_clean_max:
            ts_clean_max = ts

    # flush if buffer large
    if len(buf_user) >= BATCH_ROWS:
        flush_listens()

    # per-user noise check
    max_hourly = max((len(v) for v in hour_distinct.values()), default=0)
    max_daily = max((max(v.values()) for v in day_rec_count.values()), default=0)
    if max_hourly > NOISE_HOURLY:
        flood_flagged[user_int] = max_hourly
    if max_daily > NOISE_DAILY_REPEAT:
        repeat_flagged[user_int] = max_daily

    if has_clean:
        users_retained += 1
    else:
        users_dropped += 1

    if (user_int + 1) % 2000 == 0:
        log(f"  {user_int + 1}/{len(user_members)} users, {clean_count:,} clean listens so far")

flush_listens()
listen_writer.close()
tf.close()

# delete tar from scratch (NOT persisted in output — re-downloadable, md5-verified)
tar_path.unlink(missing_ok=True)
log(f"  tar deleted from {SCRATCH}")

log(f"  pass done: {raw_listens:,} raw → {clean_count:,} clean "
    f"({users_retained:,} retained, {users_dropped:,} dropped)")


# ════════════════════════════════════════════════════════════════════════
# STEP C: finalize outputs
# ════════════════════════════════════════════════════════════════════════
log("STEP C: writing users.parquet")
users_schema = pa.schema([("user_int", pa.int32()), ("user_uuid", pa.string())])
users_pq = CLEAN / "users.parquet"
u_ints = [u[0] for u in user_rows]
u_uuids = [u[1] for u in user_rows]
tbl = pa.table({"user_int": pa.array(u_ints, pa.int32()),
                "user_uuid": pa.array(u_uuids, pa.string())})
pq.write_table(tbl, str(users_pq), compression="zstd")
log(f"  users.parquet: {len(user_rows):,} users → {gb(users_pq.stat().st_size)}")


# ════════════════════════════════════════════════════════════════════════
# STEP D: reports
# ════════════════════════════════════════════════════════════════════════
log("STEP D: writing reports")

# --- stats ---
counts = sorted(user_listen_counts.values())
def pct(p):
    return counts[min(len(counts) - 1, int(len(counts) * p))] if counts else 0

user_stats = {
    "mean": round(sum(counts) / len(counts), 1) if counts else 0,
    "median": pct(0.5), "p90": pct(0.9),
    "min": min(counts) if counts else 0,
    "max": max(counts) if counts else 0,
}

recording_pct = round(recording_mbid_count / clean_count * 100, 1) if clean_count else 0
total_rejected = sum(rejected.values())

# noise
flood_flagged_count = len(flood_flagged)
repeat_flagged_count = len(repeat_flagged)
top_flood = sorted(flood_flagged.items(), key=lambda x: -x[1])[:10]
top_repeat = sorted(repeat_flagged.items(), key=lambda x: -x[1])[:10]

# file sizes
sizes = {}
for p in sorted(CLEAN.glob("*.parquet")):
    sizes[p.name] = gb(p.stat().st_size)

hw = {"platform": platform.platform(), "python": platform.python_version(),
      "cpus": os.cpu_count()}

# rejection rows
rej_rows = "".join(
    f"| {r} | {c:,} | {round(c / raw_listens * 100, 2) if raw_listens else 0}% |\n"
    for r, c in sorted(rejected.items(), key=lambda x: -x[1])
)
rej_ex_rows = "\n\n".join(
    f"**{r}**\n```\n" + "\n".join(repr(x) for x in exs) + "\n```"
    for r, exs in sorted(rej_examples.items())
)

# noise flood top-10
flood_rows = "".join(f"| {ui} | {v} |\n" for ui, v in top_flood)
repeat_rows = "".join(f"| {ui} | {v} |\n" for ui, v in top_repeat)
size_rows = "".join(f"| {n} | {s} |\n" for n, s in sizes.items())

(REP / "data_quality_report.md").write_text(f"""# Data Quality Report — MLHD+ Complete Shard Sanitization

**Generated:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}
**Kernel:** lb_sanitize.py (Phase 2 — complete shard)

**Shard note:** This is a *complete* shard (`mlhdplus-complete-f`). Rows are resolved
by MetaBrainz; recording MBID is expected ~100% by construction. Measured: **{recording_pct}%**.

## Input Summary

| Item | Value |
|---|---|
| Raw listens | {raw_listens:,} |
| Users (files) | {users_total:,} |
| Expected scale | ~2.5B rows |
| Source URL | {MLHD_URL} |
| Source MD5 | `{MLHD_MD5}` |
| Source bytes | {MLHD_BYTES:,} ({gb(MLHD_BYTES)}) |
| Raw tar persisted | **No** — deleted after pass; re-downloadable + md5-verified |

## Column Semantics (corrected per musicbrainz.org/doc/MLHD%2B)

| Column | Field | Notes |
|---|---|---|
| col0 | unix timestamp (sec) | validated: {TS_MIN}..NOW+{FUTURE_SLOP} |
| col1 | artist_mbids | comma-separated UUIDs (may be empty) |
| col2 | release_mbid | uuid or empty |
| col3 | recording_mbid | uuid or empty (expected ~100% in complete shard) |

No MSIDs in this dataset. No mapping join required.

## Rejections

| Reason | Count | % of raw |
|---|---|---|
{rej_rows}| **Total rejected** | **{total_rejected:,}** | **{round(total_rejected / raw_listens * 100, 2) if raw_listens else 0}%** |

### Example rejected lines (up to 3 per reason)

{rej_ex_rows}

## Kept-row Flags

| Flag | Count | Description |
|---|---|---|
| malformed artist | {malformed_artist_count:,} | artist col non-empty but not uuid(,uuid)* — stat only, artist not stored |
| malformed release | {malformed_release_count:,} | release non-empty but not valid uuid (row kept, release nulled) |

## Deduplication

| Metric | Count |
|---|---|
| Duplicates removed | {duplicates_removed:,} |

## Clean Output

| Metric | Value |
|---|---|
| Clean listens | {clean_count:,} |
| Users retained | {users_retained:,} |
| Users dropped (all rows invalid) | {users_dropped:,} |
| Per-user mean | {user_stats['mean']} |
| Per-user median | {user_stats['median']:,} |
| Per-user p90 | {user_stats['p90']:,} |
| Per-user min/max | {user_stats['min']:,} / {user_stats['max']:,} |
| Timestamp range | {ts_clean_min} .. {ts_clean_max} |

## Recording MBID Coverage

| Metric | Value |
|---|---|
| Rows with valid recording_mbid | {recording_mbid_count:,} ({recording_pct}%) |
| Expected (complete shard) | ~100% |

## Noise Flags

| Metric | Value |
|---|---|
| Users flagged flood (≥{NOISE_HOURLY}/hr) | {flood_flagged_count} |
| Users flagged repeat (≥{NOISE_DAILY_REPEAT}/day) | {repeat_flagged_count} |

**Note:** flagged, not removed. Repeat-day counts same *recording* >{NOISE_DAILY_REPEAT}/day.

### Top 10 flood-hour users
| User int | Max recordings/hour |
|---|---|
{flood_rows}
### Top 10 repeat-day users
| User int | Max same-recording/day |
|---|---|
{repeat_rows}
## Output File Sizes

| File | Size |
|---|---|
{size_rows}
## Remaining Known Issues

1. **Noise thresholds:** {NOISE_HOURLY}/hr and {NOISE_DAILY_REPEAT}/day are heuristic knobs. Flagged users are retained; thresholds may need tuning.
2. **Dedup key:** (user, timestamp, recording_mbid). Same user + same second + same recording = duplicate. Cross-user duplicates expected and retained.
3. **Raw tar not persisted:** Deleted from scratch after pass. Re-downloadable from source with MD5 verification.
4. **Power-user bias (inherited from MLHD):** source dataset filtered to users with >=2 years activity and >=10 scrobbles/day (Vigliensoni & Fujinaga, ISMIR 2017) — over-represents heavy listeners. Mitigate at training time (per-user weighting), not here.
5. **Bots / duplicate accounts:** original MLHD did not filter scripted accounts; our flood/repeat flags are the detection mechanism — decide exclusion/downweighting at training time.
6. **Provenance:** MLHD root data was scraped from Last.fm (undocumented API method, ToS-era consent); MLHD+ is republished by MetaBrainz under CC0. Recorded in DATASET_LICENSES.md.
""")

(WORK / "README.md").write_text(f"""# Music Recommender — Phase 2 (Sanitization Kernel — Complete Shard)

Sanitizes the MLHD+ *complete* shard (`mlhdplus-complete-f`): validates timestamps
and IDs, deduplicates, detects noise users, writes clean parquet.

**Input:** MLHD+ complete shard (`mlhdplus-complete-f.tar`) — no mapping needed (complete shard = MetaBrainz-resolved, recording MBID expected ~100%)
**Output:** `data/clean/listens.parquet`, `data/clean/users.parquet`
**Scratch:** tar downloads to `/kaggle/tmp/` (outside working quota), deleted after pass

Rerun: Kaggle kernel, internet enabled, CPU. Download MD5-verified. Reports deterministic.
""")

summary = {
    "shard": "mlhdplus-complete-f",
    "raw_listens": raw_listens, "users_total": users_total,
    "clean_listens": clean_count, "users_retained": users_retained,
    "users_dropped": users_dropped,
    "rejected": dict(rejected), "total_rejected": total_rejected,
    "malformed_artist_count": malformed_artist_count,
    "malformed_release_count": malformed_release_count,
    "duplicates_removed": duplicates_removed,
    "recording_mbid_count": recording_mbid_count, "recording_pct": recording_pct,
    "ts_min": ts_clean_min, "ts_max": ts_clean_max,
    "noise_flood_flagged": flood_flagged_count, "noise_repeat_flagged": repeat_flagged_count,
    "user_stats": user_stats,
    "file_sizes": sizes, "hardware": hw,
}
(WORK / "summary.json").write_text(json.dumps(summary, indent=1))
log("DONE — all reports written")
print(json.dumps({k: summary[k] for k in
      ("shard", "raw_listens", "clean_listens", "users_retained", "users_dropped",
       "duplicates_removed", "recording_pct")}, indent=1))
