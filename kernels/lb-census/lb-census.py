"""
Cross-shard census kernel: lightweight scan of MLHD+ shards (review R3-P2).

For each audited shard (streaming download -> MD5 verify -> scan -> delete):
  users (= files), rows (= raw lines), per-user ts span (first/last line),
  empty-recording-column lines (byte-level coverage proxy).

Purpose: validate that complete-tier shards other than the audited complete-f
also carry ~100% recording identifiers, and that shard-level statistics
generalize across the 16-shard split. Census counts RAW lines; it does not
re-run validation/dedup (the sanitize kernel's job), so counts may differ
from sanitize outputs by the (tiny) rejected-event margin.

Notes:
  - Files are per-user and ts-ordered, so first/last line give the ts span;
    interior lines are not parsed (census, not validation).
  - Empty-recording detection is the byte-level proxy b"\\t\\n" (line ends
    with an empty 4th column), matching MLHD+'s fixed 4-column format.
  - Partial tier is SAMPLED (4 of 16 shards: 0, 5, a, f): the partial tier's
    lack of recording MBIDs is documented upstream; the sample adds empirical
    evidence without doubling runtime.
  - Time-budget guard: shards not started before TIME_BUDGET_S elapses are
    reported as skipped.
"""
import json, sys, time, hashlib, tarfile, urllib.request
from pathlib import Path

BASE = "https://data.metabrainz.org/pub/musicbrainz/listenbrainz/mlhd"
WORK = Path("/kaggle/working"); REP = WORK / "reports"
SCRATCH = Path("/kaggle/tmp")
COMPLETE_SHARDS = list("0123456789abcdef")  # upstream layout: 0-9 + a-f (16 shards)
PARTIAL_SAMPLE = ["0", "5", "a", "f"]      # sampled partial-tier shards (same layout)
TIME_BUDGET_S = 10.5 * 3600                   # stop starting new shards after this

try:
    import zstandard as zstd
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zstandard"], check=True)
    import zstandard as zstd

T0 = time.time()
def log(m): print(f"[{int(time.time()-T0):>6}s] {m}", flush=True)
def gb(n): return f"{n/1e9:.2f} GB"

# ---------------- per-file scan ----------------

def scan_buffer(buf):
    """Return (rows, empty_recording, ts_first, ts_last) for one user file."""
    if not buf:
        return 0, 0, None, None
    rows = buf.count(b"\n")
    ends_nl = buf.endswith(b"\n")
    if not ends_nl:
        rows += 1
    empty = buf.count(b"\t\n")
    if not ends_nl and buf.rstrip(b"\r").endswith(b"\t"):
        empty += 1  # last line lacks \n but ends with empty recording col
    ts_first = ts_last = None
    try:
        ts_first = int(buf.split(b"\n", 1)[0].split(b"\t", 1)[0])
    except ValueError:
        pass
    try:
        ts_last = int(buf.rstrip(b"\n").rsplit(b"\n", 1)[-1].split(b"\t", 1)[0])
    except ValueError:
        pass
    return rows, empty, ts_first, ts_last

# ---------------- per-shard census ----------------

def fetch_md5s():
    with urllib.request.urlopen(f"{BASE}/md5sums", timeout=60) as r:
        text = r.read().decode("utf-8", "replace")
    m = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            m[parts[1].lstrip("*")] = parts[0].strip()
    return m

def download_and_verify(name, expected):
    dest = SCRATCH / name
    h = hashlib.md5()
    t0 = time.time()
    with urllib.request.urlopen(f"{BASE}/{name}", timeout=600) as r, open(dest, "wb") as f:
        done = 0
        while True:
            chunk = r.read(1 << 23)
            if not chunk:
                break
            f.write(chunk); h.update(chunk); done += len(chunk)
    if expected and h.hexdigest() != expected:
        raise RuntimeError(f"MD5 mismatch for {name}: {h.hexdigest()} != {expected}")
    return dest, done, time.time() - t0

def census_tar(path):
    """Scan every .txt.zst member; return aggregate counts."""
    dctx = zstd.ZstdDecompressor()
    users = rows = empty = bad_span = 0
    ts_min = ts_max = None
    t0 = time.time()
    with tarfile.open(path, "r:") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".txt.zst"):
                continue
            with tf.extractfile(m) as fh:
                buf = dctx.stream_reader(fh).read()
            r_, e_, f_, l_ = scan_buffer(buf)
            users += 1; rows += r_; empty += e_
            if f_ is None or l_ is None:
                bad_span += 1
            else:
                ts_min = f_ if ts_min is None else min(ts_min, f_)
                ts_max = l_ if ts_max is None else max(ts_max, l_)
    return dict(users=users, rows=rows, empty_recording_lines=empty,
                bad_span_files=bad_span, ts_min=ts_min, ts_max=ts_max,
                scan_seconds=round(time.time() - t0))

def run_census():
    REP.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    log("fetching official md5sums")
    md5map = fetch_md5s()
    plan = [("complete", s) for s in COMPLETE_SHARDS] + \
           [("partial", s) for s in PARTIAL_SAMPLE]
    results, skipped, failed = [], [], []
    for tier, letter in plan:
        name = f"mlhdplus-{tier}-{letter}.tar"
        if time.time() - T0 > TIME_BUDGET_S:
            skipped.append(name)
            continue
        log(f"{name}: downloading")
        try:
            dest, nbytes, dl_s = download_and_verify(name, md5map.get(name))
        except Exception as e:  # 404/URLError/MD5 mismatch: log, skip, continue
            log(f"{name}: FAILED ({e.__class__.__name__}: {e}) — skipping")
            failed.append(dict(shard=name, tier=tier, error=f"{e.__class__.__name__}: {e}"))
            continue
        log(f"{name}: {gb(nbytes)} in {dl_s:.0f}s (md5 ok) — scanning")
        stats = census_tar(dest)
        dest.unlink()
        stats.update(shard=name, tier=tier, dl_seconds=round(dl_s),
                     bytes=nbytes,
                     recording_coverage_pct=round(100.0 * (1 - stats["empty_recording_lines"] / stats["rows"]), 4) if stats["rows"] else None)
        results.append(stats)
        log(f"{name}: users={stats['users']:,} rows={stats['rows']:,} "
            f"empty_rec={stats['empty_recording_lines']:,} "
            f"cov={stats['recording_coverage_pct']}% "
            f"ts=[{stats['ts_min']}..{stats['ts_max']}]")
    # aggregate
    comp = [r for r in results if r["tier"] == "complete"]
    part = [r for r in results if r["tier"] == "partial"]
    agg = {}
    for label, rs in (("complete", comp), ("partial", part)):
        if rs:
            agg[label] = dict(
                shards=len(rs), users=sum(r["users"] for r in rs),
                rows=sum(r["rows"] for r in rs),
                empty_recording_lines=sum(r["empty_recording_lines"] for r in rs),
                ts_min=min(r["ts_min"] for r in rs if r["ts_min"] is not None),
                ts_max=max(r["ts_max"] for r in rs if r["ts_max"] is not None),
                coverage_min=min(r["recording_coverage_pct"] for r in rs),
                coverage_max=max(r["recording_coverage_pct"] for r in rs))
    summary = dict(scanned=results, skipped_for_time_budget=skipped,
                   failed_shards=failed,
                   aggregate=agg, time_budget_s=TIME_BUDGET_S,
                   total_seconds=round(time.time() - T0))
    (WORK / "census_summary.json").write_text(json.dumps(summary, indent=1))
    # report
    lines = ["# MLHD+ Cross-Shard Census", "",
             "Lightweight raw-line census (no validation/dedup; see sanitize kernel for that).",
             "Empty-recording counts are the byte-level proxy `TAB + newline` (empty 4th column).", ""]
    for tier_label, rs in (("Complete tier (all 16 shards)", comp),
                           ("Partial tier (sampled 4 of 16)", part)):
        lines += [f"## {tier_label}", "",
                  "| shard | users | rows | empty rec | coverage % | ts min | ts max |",
                  "|---|---|---|---|---|---|---|"]
        for r in rs:
            lines.append(f"| {r['shard']} | {r['users']:,} | {r['rows']:,} | "
                         f"{r['empty_recording_lines']:,} | {r['recording_coverage_pct']} | "
                         f"{r['ts_min']} | {r['ts_max']} |")
        lines.append("")
    if agg:
        lines += ["## Aggregate", ""]
        for k, v in agg.items():
            lines.append(f"- **{k}**: {v['shards']} shards, {v['users']:,} users, "
                         f"{v['rows']:,} rows, coverage {v['coverage_min']}–{v['coverage_max']}%, "
                         f"ts [{v['ts_min']}..{v['ts_max']}]")
        lines.append("")
    if skipped:
        lines += [f"**Skipped for time budget:** {', '.join(skipped)}", ""]
    if failed:
        lines += ["## Failed shards", ""]
        for f_ in failed:
            lines.append(f"- {f_['shard']}: {f_['error']}")
        lines.append("")
    lines += ["## Notes", "",
              "- Counts are raw lines; sanitize rejects a tiny margin (2 events in complete-f),",
              "  so census rows may exceed sanitize-retained rows slightly.",
              "- ts span uses first/last line per file (files are ts-ordered per user).",
              "- Partial tier sampled (0, 5, a, f): its missing recording MBIDs are documented",
              "  upstream; the sample adds cross-shard empirical evidence only.",
              ""]
    (REP / "census_report.md").write_text("\n".join(lines))
    log(f"DONE — {len(results)} shards scanned, {len(skipped)} skipped")
    print(json.dumps({k: summary["aggregate"][k] for k in summary["aggregate"]}))

# ---------------- synthetic test ----------------

def _synthetic_test():
    import tempfile, tarfile as tt
    import zstandard as zstd
    d = Path(tempfile.mkdtemp())
    cctx = zstd.ZstdCompressor()
    u1 = b"100\tart1\trel1\trec1\n101\tart1\trel1\trec1\n102\tart1\trel1\trec1\n103\tart1\trel1\trec1\n104\tart1\trel1\trec1\n"
    u2 = b"200\tart2\trel2\trec2\n201\tart2\trel2\trec2\n202\tart2\trel2\t\n203\tart2\trel2\t"  # 2 empty rec; no trailing \n; last ends with empty rec
    u3 = b"300\tart3\trel3\trec3\n"
    tar_path = d / "mini.tar"
    with tt.open(tar_path, "w") as tf:
        for name, data in (("00/u1.txt.zst", u1), ("00/u2.txt.zst", u2), ("01/u3.txt.zst", u3)):
            zdata = cctx.compress(data)
            member = tt.TarInfo(name); member.size = len(zdata)
            import io
            tf.addfile(member, io.BytesIO(zdata))
    stats = census_tar(tar_path)
    assert stats["users"] == 3, stats
    assert stats["rows"] == 10, stats                      # 5 + 4 + 1
    assert stats["empty_recording_lines"] == 2, stats      # u2 lines 3,4 (one via \t\n, one via no-newline tail)
    assert stats["ts_min"] == 100 and stats["ts_max"] == 300, stats
    assert stats["bad_span_files"] == 0, stats
    # scan_buffer edge: empty buffer
    assert scan_buffer(b"") == (0, 0, None, None)
    # md5 map parse
    print("ALL CENSUS TESTS PASSED", stats)

if __name__ == "__main__":
    if "--test" in sys.argv:
        _synthetic_test()
    else:
        run_census()
