"""
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Shuvi

Eyeball sanity-check kernel: verify item2vec embeddings semantically.

Anchors = 20 most-listened recordings (vocab is count-sorted, item_id 0..19).
For each: top-10 cosine neighbors, resolved to artist/title via MusicBrainz.
Writes a human-readable neighbor table + same-artist rate.

Input:  item2vec_final.npy (lb-item2vec) + vocab.parquet (lb-trainprep)
Output: reports/eyeball_report.md, eyeball_summary.json, README.md
CPU + internet-enabled. ~220 MB API calls at 1.1s spacing = 4-5 min.
"""
import json, os, sys, time, uuid
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────
N_ANCHORS  = 20
TOP_K      = 10
MB_SLEEP   = 1.1       # seconds between MusicBrainz requests (rate limit)
MB_RETRIES = 2
MB_TIMEOUT = 15

import numpy as np
import pyarrow.parquet as pq

T0 = time.time()
def log(m): print(f"[{int(time.time()-T0):>5}s] {m}", flush=True)


def mbid_str(b):
    """16-byte binary MBID -> UUID string for the MusicBrainz API."""
    return str(uuid.UUID(bytes=b))


def find_neighbors(emb_norm, idx, k):
    """(sim, item_id) pairs of k nearest neighbors of idx, best first; self excluded."""
    sims = emb_norm @ emb_norm[idx]
    sims[idx] = -2.0
    k = min(k, len(sims) - 1)
    top = np.argpartition(-sims, k - 1)[:k]
    order = top[np.argsort(-sims[top])]
    return [(float(sims[i]), int(i)) for i in order]


def _mb_get(url):
    import urllib.request
    for attempt in range(MB_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "music-rec-engine-eyeball/0.1 (kaggle: shuvitobe)"})
            with urllib.request.urlopen(req, timeout=MB_TIMEOUT) as resp:
                return json.loads(resp.read())
        except Exception:
            if attempt < MB_RETRIES:
                time.sleep(MB_SLEEP * (attempt + 1))
    return None


def lookup_recording(mbid_b):
    """(artist, title) for a recording MBID, or (None, None) on failure/404."""
    url = f"https://musicbrainz.org/ws/2/recording/{mbid_str(mbid_b)}?inc=artist-credits&fmt=json"
    time.sleep(MB_SLEEP)
    data = _mb_get(url)
    if not data or not isinstance(data, dict):
        return (None, None)
    credits = data.get("artist-credit") or []
    artist = credits[0].get("name") if credits and isinstance(credits[0], dict) else None
    return (artist, data.get("title"))


def run_eyeball(emb_path, vocab_path, out_dir):
    log(f"Loading embeddings: {emb_path}")
    emb = np.load(str(emb_path))
    log(f"  shape: {emb.shape}")

    vt = pq.read_table(str(vocab_path), columns=["recording_mbid", "count"])
    mbids = vt.column("recording_mbid").to_pylist()
    counts = vt.column("count").to_pylist()
    n = len(mbids)
    log(f"  vocab rows: {n}")
    assert emb.shape[0] == n, f"emb rows ({emb.shape[0]}) != vocab rows ({n})"

    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

    # Anchors = most-listened recordings (vocab sorted by count desc)
    anchor_idx = list(range(min(N_ANCHORS, n)))
    log(f"  anchors: item_id 0..{len(anchor_idx)-1} (most-listened)")

    neighbor_map = {ai: find_neighbors(emb_norm, ai, TOP_K) for ai in anchor_idx}

    resolve_set = set(anchor_idx)
    for nbs in neighbor_map.values():
        resolve_set.update(i for _, i in nbs)
    resolve_list = sorted(resolve_set)
    log(f"  unique MBIDs to resolve: {len(resolve_list)}")

    info = {}
    n_404 = 0
    for i, idx in enumerate(resolve_list):
        if i % 20 == 0:
            log(f"  MB queries {i}/{len(resolve_list)}")
        info[idx] = lookup_recording(mbids[idx])
        if info[idx] == (None, None):
            n_404 += 1
    n_resolved = len(resolve_list) - n_404
    log(f"  resolved: {n_resolved}, 404/failed: {n_404}")

    def name(idx):
        a, t = info.get(idx, (None, None))
        if a and t: return f"{a} - {t}"
        if t: return t
        return f"[unresolved {mbid_str(mbids[idx])[:8]}]"

    # same-artist rate across anchor-neighbor pairs
    same = total = 0
    for ai in anchor_idx:
        a_artist = info.get(ai, (None, None))[0]
        if a_artist is None:
            continue
        for _, ni in neighbor_map[ai]:
            n_artist = info.get(ni, (None, None))[0]
            if n_artist is not None:
                total += 1
                if a_artist == n_artist:
                    same += 1
    rate = same / total if total else 0.0
    log(f"  same-artist: {same}/{total} = {rate:.3f}")

    # ── outputs ──────────────────────────────────────────────────────
    out = Path(out_dir)
    rep_dir = out / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    lines = ["# Item2Vec Eyeball Sanity Check", "",
             f"Anchors = {len(anchor_idx)} most-listened recordings; top-{TOP_K} "
             f"cosine neighbors each. Same-artist rate: **{rate:.3f}** "
             f"({same}/{total}).", "",
             "**Definitions:** the denominator ({}) counts anchor-neighbor "
             "*pairs* whose artist resolved on both sides; {} of those pairs "
             "share an artist. `n_resolved` ({}) counts unique recordings "
             "resolved via the MusicBrainz API (anchors and neighbors "
             "overlap, so it is not the sum of the two groups); `n_404` "
             "counts API 404s (deleted/merged recordings). This is a "
             "qualitative embedding sanity check, not a model "
             "evaluation.".format(total, same, n_resolved), ""]
    for ai in anchor_idx:
        lines.append(f"## {name(ai)}  ({counts[ai]:,} listens)")
        lines.append("")
        lines.append("| rank | neighbor | cosine |")
        lines.append("|---|---|---|")
        for r, (s, ni) in enumerate(neighbor_map[ai], 1):
            lines.append(f"| {r} | {name(ni)} | {s:.3f} |")
        lines.append("")
    (rep_dir / "eyeball_report.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {rep_dir / 'eyeball_report.md'}")

    summary = dict(same_artist_rate=round(rate, 4), same_artist_count=same,
                   total_pairs=total, n_anchors=len(anchor_idx),
                   n_resolved=n_resolved, n_404=n_404, top_k=TOP_K)
    (out / "eyeball_summary.json").write_text(json.dumps(summary, indent=1),
                                               encoding="utf-8")
    (out / "README.md").write_text(
        "# Item2Vec Eyeball Sanity Check\n\n"
        "Anchors = top-20 most-listened recordings; top-10 cosine neighbors\n"
        "resolved via MusicBrainz. See reports/eyeball_report.md and\n"
        "eyeball_summary.json. Same-artist rate should be well above chance\n"
        "but below 100% (both extremes suspicious).\n", encoding="utf-8")
    print(json.dumps(summary))
    return summary


def main():
    root = Path("/kaggle/input")
    tree = "\n".join(f"  {p}" for p in sorted(root.rglob("*")) if p.is_file())
    log(f"/kaggle/input tree:\n{tree}")

    emb_files = list(root.rglob("item2vec_final.npy"))
    vocab_files = list(root.rglob("vocab.parquet"))
    if not emb_files:
        sys.exit("FATAL: item2vec_final.npy not found under /kaggle/input")
    if len(emb_files) > 1:
        sys.exit(f"FATAL: multiple item2vec_final.npy found, expected exactly one: {emb_files}")
    if not vocab_files:
        sys.exit("FATAL: vocab.parquet not found under /kaggle/input")
    if len(vocab_files) > 1:
        sys.exit(f"FATAL: multiple vocab.parquet found, expected exactly one: {vocab_files}")

    log(f"Embeddings: {emb_files[0]}")
    log(f"Vocab:      {vocab_files[0]}")
    run_eyeball(emb_files[0], vocab_files[0], Path("/kaggle/working"))


if __name__ == "__main__":
    main()
