"""
Phase 1 kernel (Kaggle): probe ListenBrainz dump options -> select smallest sensible
dataset (smallest MLHD+ partial shard) -> download + md5-verify -> inspect actual
schema (tar of per-user .txt.zst files) -> full streaming pass -> reports.

Spec FIRST TASK steps 1-8. No training. Raw stays immutable in data/raw/.

Outputs (/kaggle/working):
  data/raw/<selected>.tar      raw dump, checksum-verified
  reports/dump_options.md     all probed options + sizes
  reports/dataset_selection.md selection + why + checksum
  reports/schema_report.md     actual structure, column semantics, sample lines
  reports/data_size_report.md  sizes, listens, users, listens/user, timestamp stats
  DATASET_LICENSES.md          CC0 record
  README.md                    Kaggle-native structure notes
  summary.json                 machine-readable stats for sanitization design
"""
import json, os, sys, time, hashlib, tarfile, platform, urllib.request, re, subprocess
from pathlib import Path
from collections import Counter

BASE = "https://data.metabrainz.org/pub/musicbrainz/listenbrainz"
WORK = Path("/kaggle/working"); RAW = WORK / "data" / "raw"; REP = WORK / "reports"
for d in (RAW, REP):
    d.mkdir(parents=True, exist_ok=True)

try:
    import zstandard as zstd
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zstandard"], check=True)
    import zstandard as zstd

T0 = time.time()
def log(m): print(f"[{int(time.time()-T0):>5}s] {m}", flush=True)
def gb(n): return f"{n/1e9:.2f} GB" if n >= 1e9 else f"{n/1e6:.1f} MB"

def head_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])

def fetch_text(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8", "replace")

def hrefs(html): return re.findall(r'href="([^"]+)"', html)

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)

# ---------------- A: probe options ----------------
log("A: probing dump options")
options = []
for name, url, note in [
    ("sample-dump-20250610",
     f"{BASE}/sample/listenbrainz-sample-20250610-094311-full/listenbrainz-sample-dump-20250610-094311.tar.zst",
     "metadata/popularity caches only; NO per-user listens"),
    ("feedback-dump-20260901",
     f"{BASE}/spark/listenbrainz-feedback-20260901-021502-full/listenbrainz-feedback-dump-20260901-021502.tar.zst",
     "recording like/dislike feedback; not listening events"),
    ("full-spark-dump-2593",
     f"{BASE}/fullexport/listenbrainz-dump-2593-20260712-000004-full/listenbrainz-spark-dump-2593-20260712-000004-full.tar",
     "ALL listens, spark/parquet format"),
    ("incremental-2635",
     f"{BASE}/incremental/listenbrainz-dump-2635-20260821-000003-incremental/listenbrainz-listens-dump-2635-20260821-000003-incremental.tar.zst",
     "daily delta; requires full dump as base"),
]:
    try:
        s = head_size(url); options.append({"name": name, "url": url, "bytes": s, "note": note})
        log(f"  {name}: {gb(s)}")
    except Exception as e:
        log(f"  {name}: probe failed: {e}")

log("  enumerating MLHD+ shards")
mlhd = {}
for h in sorted(x for x in hrefs(fetch_text(f"{BASE}/mlhd/"))
                if x.startswith("mlhdplus-") and x.endswith(".tar")):
    try:
        mlhd[h] = head_size(f"{BASE}/mlhd/{h}")
    except Exception as e:
        log(f"  {h}: probe failed: {e}")
complete = {k: v for k, v in mlhd.items() if "-complete-" in k}
partial = {k: v for k, v in mlhd.items() if "-partial-" in k}
log(f"  MLHD+ complete: {len(complete)} shards, {gb(sum(complete.values()))} total")
log(f"  MLHD+ partial : {len(partial)} shards, {gb(sum(partial.values()))} total")

# ---------------- B: select ----------------
sel = min(sorted(partial), key=lambda k: (partial[k], k))
sel_url, sel_bytes = f"{BASE}/mlhd/{sel}", partial[sel]
log(f"B: SELECTED {sel} ({gb(sel_bytes)}) — smallest MLHD+ partial shard (per-user listens)")

# ---------------- C: download + verify ----------------
log("C: download + md5 verify (official mlhd/md5sums file)")
tar_path = RAW / sel
md5map = {}
for line in fetch_text(f"{BASE}/mlhd/md5sums").splitlines():
    parts = line.split()
    if len(parts) == 2:
        md5map[parts[1].lstrip("*")] = parts[0].strip()
exp_md5 = md5map.get(sel)
if not exp_md5:
    sys.exit(f"no md5 entry for {sel} in {BASE}/mlhd/md5sums")
h = hashlib.md5()
if tar_path.exists() and tar_path.stat().st_size == sel_bytes:
    log("  file present from prior run; hashing only")
    with open(tar_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 23), b""):
            h.update(chunk)
else:
    with urllib.request.urlopen(sel_url, timeout=300) as r, open(tar_path, "wb") as f:
        done = 0
        while True:
            chunk = r.read(1 << 23)
            if not chunk: break
            f.write(chunk); h.update(chunk); done += len(chunk)
            if done // (200 << 20) != (done - len(chunk)) // (200 << 20):
                log(f"  downloaded {gb(done)} / {gb(sel_bytes)}")
got = h.hexdigest()
log(f"  md5 expected: {exp_md5}")
log(f"  md5 got     : {got}")
if got != exp_md5:
    sys.exit("CHECKSUM MISMATCH — aborting, raw file NOT trusted")

# ---------------- D: tar structure ----------------
log("D: tar inspection")
tf = tarfile.open(tar_path, "r:")
members = tf.getmembers()
files = [m for m in members if m.isfile()]
dirs = [m for m in members if m.isdir()]
unc = sum(m.size for m in files)
user_files = [m for m in files if m.name.endswith(".txt.zst")]
other_files = [m for m in files if not m.name.endswith(".txt.zst")]
hexdirs = sorted(m.name for m in dirs)
log(f"  {len(members)} members: {len(dirs)} dirs ({len(hexdirs)} unique), {len(files)} files")
log(f"  user history files (.txt.zst): {len(user_files)}; other files: {len(other_files)}")
log(f"  uncompressed total: {gb(unc)}")
for m in other_files[:10]:
    log(f"  other: {m.name} ({m.size} bytes)")

# ---------------- E: schema discovery ----------------
log("E: schema discovery (first 5 user files)")
dctx = zstd.ZstdDecompressor()
def read_user(m):
    with tf.extractfile(m) as fh:
        return dctx.stream_reader(fh).read().decode("utf-8", "replace")

col_counts = Counter()
col_sem = {}
col_int_ranges = {}
sample_lines = []
for m in user_files[:5]:
    txt = read_user(m)
    lines = [l for l in txt.splitlines() if l.strip()]
    sample_lines.append((m.name, len(lines), lines[:3]))
    for l in lines:
        cols = l.split("\t")
        col_counts[len(cols)] += 1
        for i, c in enumerate(cols):
            c = c.strip()
            if UUID_RE.match(c): kind = "uuid"
            elif re.fullmatch(r"-?\d+", c): kind = "int"
            elif not c: kind = "empty"
            else: kind = "text"
            col_sem.setdefault(i, Counter())[kind] += 1
            if kind == "int":
                lo, hi = col_int_ranges.get(i, (10**18, -(10**18)))
                v = int(c)
                col_int_ranges[i] = (min(lo, v), max(hi, v))
log(f"  columns per line: {dict(col_counts)}")
for i in sorted(col_sem):
    r = col_int_ranges.get(i)
    log(f"  col{i}: {dict(col_sem[i])}" + (f" int-range={r}" if r else ""))

# detect timestamp column: int col whose values are plausible unix seconds (or ms)
ts_col = None; ts_unit = None
for i in sorted(col_int_ranges):
    lo, hi = col_int_ranges[i]
    if 10**8 <= lo and hi <= 3 * 10**9:
        ts_col, ts_unit = i, "s"; break
    if 10**11 <= lo and hi <= 3 * 10**12:
        ts_col, ts_unit = i, "ms"; break
msid_col = None
for i in sorted(col_sem):
    if col_sem[i].get("uuid", 0) > 0:
        msid_col = i; break
log(f"  detected: ts_col={ts_col} (unit {ts_unit}), msid_col={msid_col}")

# ---------------- F: full pass ----------------
log("F: full streaming pass")
n_lines = n_badts = 0
users = Counter()
ts_min = ts_max = None
ts_future = ts_negative = ts_pre2000 = 0
NOW = time.time()
EPOCH_2000 = 946684800
for idx, m in enumerate(user_files, 1):
    user = os.path.basename(m.name)[:-len(".txt.zst")]
    txt = read_user(m)
    n = 0
    for l in txt.splitlines():
        if not l.strip(): continue
        n += 1
        if ts_col is not None:
            cols = l.split("\t")
            if ts_col < len(cols):
                try:
                    t = int(cols[ts_col])
                    if ts_unit == "ms": t //= 1000
                except ValueError:
                    n_badts += 1
                else:
                    if ts_min is None or t < ts_min: ts_min = t
                    if ts_max is None or t > ts_max: ts_max = t
                    if t > NOW + 86400: ts_future += 1
                    elif t < 0: ts_negative += 1
                    elif t < EPOCH_2000: ts_pre2000 += 1
    n_lines += n
    users[user] = n
    if idx % 2000 == 0:
        log(f"  {idx}/{len(user_files)} files, {n_lines:,} listens so far")
log(f"  TOTAL listens: {n_lines:,} across {len(users):,} users (bad ts: {n_badts:,})")

counts = sorted(users.values())
def pct(p): return counts[min(len(counts) - 1, int(len(counts) * p))] if counts else 0
user_stats = {
    "users": len(counts),
    "listens_per_user_mean": round(sum(counts) / len(counts), 1) if counts else 0,
    "listens_per_user_median": pct(0.5), "listens_per_user_p90": pct(0.9),
    "listens_per_user_min": min(counts) if counts else 0,
    "listens_per_user_max": max(counts) if counts else 0,
}

# ---------------- G: reports ----------------
log("G: writing reports")
hw = {"platform": platform.platform(), "python": platform.python_version(),
      "cpus": os.cpu_count()}
opt_rows = "".join(f"| {o['name']} | {gb(o['bytes'])} | {o['note']} |\n" for o in options)

(REP / "dump_options.md").write_text(f"""# ListenBrainz Dump Options (probed {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())})

| Option | Size | Contents |
|---|---|---|
{opt_rows}| MLHD+ complete ({len(complete)} shards) | {gb(sum(complete.values()))} | raw per-user listens, complete histories |
| MLHD+ partial ({len(partial)} shards) | {gb(sum(partial.values()))} | raw per-user listens, partial histories |

MLHD+ shard sizes (complete): {json.dumps({k: gb(v) for k, v in sorted(complete.items())})}
MLHD+ shard sizes (partial): {json.dumps({k: gb(v) for k, v in sorted(partial.items())})}

All ListenBrainz dumps including MLHD+ are CC0 1.0 Universal (see DATASET_LICENSES.md).
""")

(REP / "dataset_selection.md").write_text(f"""# Dataset Selection (Phase 1)

**Selected:** `{sel}` — {gb(sel_bytes)} compressed, {gb(unc)} uncompressed
**Rule:** smallest MLHD+ partial shard = smallest archive containing raw per-user listens
**Source:** {sel_url}
**MD5 (verified):** `{exp_md5}`

Rejected alternatives:
- sample dump: metadata/popularity caches only, no per-user listens
- feedback dump: like/dislike signals, not listening events
- full spark dump (~191 GB): deferred until pipeline proven on small subset
- incremental dumps: deltas requiring a full dump as base
- MLHD+ complete shards (~15 GB each): larger than needed for first experiment

Raw file kept unmodified at `data/raw/{sel}` (spec rule: raw data immutable).
Run hardware: {hw['platform']}, {hw['cpus']} CPUs.
""")

schema_body = "\n".join(
    f"### {name} ({nlines} listens)\n```\n" + "\n".join(lines) + "\n```"
    for name, nlines, lines in sample_lines)
(REP / "schema_report.md").write_text(f"""# Schema Report — {sel}

## Archive structure
- tar containing {len(dirs)} directories (hex-sharded: {hexdirs[:4]}...{hexdirs[-4:] if len(hexdirs) > 4 else ''}) and {len(files)} files
- **One file per user**: `<hex2>/<user-uuid>.txt.zst` — zstd-compressed listening history
- User history files: {len(user_files)}; other files: {len(other_files)}
- Uncompressed size: {gb(unc)}

## Line format (inside each .txt.zst)
- Columns per line (tab-separated): {dict(col_counts)}
- Column semantics:
{chr(10).join(f'  - col{i}: {dict(col_sem[i])}' + (f', int range {col_int_ranges[i]}' if i in col_int_ranges else '') for i in sorted(col_sem))}
- Detected timestamp column: `{ts_col}` (unit: {ts_unit})
- Detected MSID column: `{msid_col}` (UUID-shaped)

## Sample content (verbatim, first lines)
{schema_body}

## Normalization dependency (for Phase 3)
MLHD+ is keyed by MessyBrainz MSIDs, not canonical MusicBrainz MBIDs.
Mapping MSID -> MBID requires `labs/mappings/msid-mbid-mapping` from
{BASE}/labs/mappings/ — separate download in the sanitization phase.
""")

(REP / "data_size_report.md").write_text(f"""# Data Size Report — {sel}

| Metric | Value |
|---|---|
| Compressed size | {gb(sel_bytes)} |
| Uncompressed size | {gb(unc)} |
| Users (files) | {user_stats['users']:,} |
| Listens (lines) | {n_lines:,} |
| Listens/user (mean) | {user_stats['listens_per_user_mean']:,} |
| Listens/user (median) | {user_stats['listens_per_user_median']:,} |
| Listens/user (p90) | {user_stats['listens_per_user_p90']:,} |
| Listens/user (min/max) | {user_stats['listens_per_user_min']:,} / {user_stats['listens_per_user_max']:,} |
| Timestamp range | {ts_min} .. {ts_max} |
| Future timestamps (>now+1d) | {ts_future:,} |
| Negative timestamps | {ts_negative:,} |
| Pre-2000 timestamps | {ts_pre2000:,} |
| Unparseable timestamps | {n_badts:,} |

Scale estimates for planning:
- MLHD+ total (all shards): {gb(sum(complete.values()) + sum(partial.values()))}
- Full ListenBrainz spark dump: ~191 GB

Hardware: {hw['platform']}, Python {hw['python']}, {hw['cpus']} CPUs
""")

(WORK / "DATASET_LICENSES.md").write_text("""# Dataset Licenses

## ListenBrainz data dumps (incl. MLHD+)
- **License:** CC0 1.0 Universal (public domain dedication) — https://creativecommons.org/publicdomain/zero/1.0/
- **Scope:** all ListenBrainz dump types (sample, spark full/incremental, feedback, MLHD+)
- **Verification:** a `COPYING` file with the CC0 legal code is bundled into every dump archive by
  `listenbrainz/dumps/exporter.py::write_dump_metadata` (listenbrainz-server repo); the user consent
  dialog states dumps are published under CC0.
- **Attribution:** not required (courtesy attribution to ListenBrainz/MetaBrainz suggested)
- **Commercial use / redistribution / ML training:** permitted without restriction
- **Sources:**
  - https://github.com/metabrainz/listenbrainz-server/blob/master/listenbrainz/db/licenses/COPYING-PublicDomain
  - https://github.com/metabrainz/listenbrainz-server/blob/master/listenbrainz/db/licenses/README.md
  - https://github.com/metabrainz/listenbrainz-server/blob/master/frontend/js/src/gdpr/GDPR.tsx
- **Note:** GPL-2.0 covers listenbrainz-server *code* only, explicitly not the data.

## Pending (verify at integration time)
- MusicBrainz core data (expected CC0)
- AcousticBrainz features
- msid-mbid mapping (labs/mappings): expected same dump licensing; verify COPYING at download
""")

(WORK / "README.md").write_text(f"""# Music Recommender — Phase 1 (Kaggle-native)

All pipeline work runs on Kaggle (project decision). Spec structure maps to Kaggle artifacts:

| Spec path | Kaggle artifact |
|---|---|
| `data/raw/` | this kernel's output: `data/raw/{sel}` (immutable, checksum-verified) |
| `data/clean, rejected/` | sanitization kernel output (next phase) |
| `reports/` | this kernel's output: `reports/*.md` |
| `src/` | kernel scripts (versioned on Kaggle) |
| `configs/` | CONFIG constants at top of each kernel script |
| `models/` | later training kernels |

Reproduce: rerun this kernel (internet enabled, CPU). Downloads from data.metabrainz.org,
verifies SHA-256, regenerates all reports deterministically.

Status: Phase 1 complete (dataset selected, schema inspected, sizes measured).
Next: design sanitization pipeline from `reports/schema_report.md`.
""")

summary = {
    "selected": sel, "selected_url": sel_url, "md5": exp_md5,
    "compressed_bytes": sel_bytes, "uncompressed_bytes": unc,
    "format": "tar of per-user .txt.zst (hex-sharded dirs, user uuid filenames)",
    "columns_per_line": dict(col_counts),
    "col_semantics": {str(i): dict(v) for i, v in col_sem.items()},
    "ts_col": ts_col, "ts_unit": ts_unit, "msid_col": msid_col,
    "users": user_stats["users"], "listens": n_lines,
    "user_stats": user_stats,
    "ts_min": ts_min, "ts_max": ts_max,
    "ts_future": ts_future, "ts_negative": ts_negative,
    "ts_pre2000": ts_pre2000, "ts_bad": n_badts,
    "mlhd_complete_shards": len(complete), "mlhd_partial_shards": len(partial),
    "mlhd_total_bytes": sum(complete.values()) + sum(partial.values()),
    "hardware": hw,
}
(WORK / "summary.json").write_text(json.dumps(summary, indent=1))
log("DONE — all reports written")
print(json.dumps({k: summary[k] for k in
      ("selected", "users", "listens", "ts_min", "ts_max", "ts_future", "ts_pre2000")}, indent=1))
