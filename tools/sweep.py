"""Configuration sweep for the hybrid ranker — all possibilities, measured.

Usage:
    python tools/sweep.py

Sweeps: tier (raw/sonata/etude) x profile size (light/medium/heavy),
plus exploration, context (peak/off-peak hour), half-life, exclude_played,
and k. Measures per-call latency and top-k overlap vs the raw tier.
Writes reports/sweep_results.json.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.context import ListenRhythm
from src.profile import LocalProfile
from src.ranker import HybridRanker

MODELS = ROOT / "models"
TIERS = {
    "raw": dict(embeddings_path=MODELS / "item2vec_final.npy"),
    "sonata": dict(pq_path=MODELS / "sonata_pq.npz"),
    "etude": dict(pq_path=MODELS / "etude_pq.npz"),
}


def build_profile(n_items: int, weighted: bool = False) -> LocalProfile:
    """Real-data profile from top_items.json (same rule as demo/benchmark)."""
    import random
    rng = random.Random(42)
    now = time.time()
    p = LocalProfile()
    top = json.loads((MODELS / "top_items.json").read_text())
    for entry in top[::6][:n_items]:
        for _ in range(rng.randint(1, 10)):
            if weighted and rng.random() < 0.3:
                p.observe(entry["v1_item_id"], now - rng.uniform(0, 90 * 86400),
                          signal=rng.choice(["like", "save", "completion"]))
            else:
                p.observe(entry["v1_item_id"], now - rng.uniform(0, 90 * 86400))
    return p


def timed_recs(ranker, profile, **kw) -> tuple[list[int], float]:
    t = time.time()
    recs = ranker.recommend(profile, **kw)
    return recs, (time.time() - t) * 1000.0


def jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


def main() -> None:
    results: dict = {"sweep_version": 1, "runs": []}

    # Profiles: light / medium / heavy / weighted-signals
    profiles = {
        "light30": build_profile(30),
        "medium100": build_profile(100),
        "heavy300": build_profile(300),
        "weighted100": build_profile(100, weighted=True),
    }
    # Rhythm learned from the medium profile's timestamps
    rhythm = ListenRhythm()
    rhythm.learn(list(profiles["medium100"].last_seen_ts.values()))
    # Peak vs off-peak hour: find argmax hour bucket, offset by 12h
    peak_hour = int(np.argmax(rhythm.hour_probs))
    now = time.time()
    import datetime as _dt
    peak_ts = now + ((_dt.datetime.fromtimestamp(now).replace(
        hour=peak_hour, minute=0, second=0) - _dt.datetime.fromtimestamp(now)).total_seconds())
    off_ts = peak_ts + 12 * 3600

    # Rankers per tier (loaded once)
    rankers = {}
    for name, kw in TIERS.items():
        r = HybridRanker(**kw)
        assert r.has_embeddings(), f"{name}: no model"
        rankers[name] = r

    def run(tag, tier, profile, **kw):
        recs, ms = timed_recs(rankers[tier], profile, **kw)
        row = {"tag": tag, "tier": tier, "ms": round(ms, 1), "top": recs[:10]}
        results["runs"].append(row)
        print(f"{tag:<38} {tier:<7} {ms:8.1f} ms  top5={recs[:5]}")
        return recs

    # ── Sweep 1: tier x profile size (defaults: k=20, excl, hl=90) ──
    print("== tier x profile ==")
    base = {}  # (tier, profile) -> recs for overlap
    for tier in TIERS:
        for pname, prof in profiles.items():
            recs = run(f"{pname}-default", tier, prof, k=20)
            base[(tier, pname)] = recs

    # Overlap vs raw, same profile
    print("\n== top-20 overlap vs raw tier ==")
    overlaps = {}
    for tier in ("sonata", "etude"):
        for pname in profiles:
            j = jaccard(base[("raw", pname)], base[(tier, pname)])
            overlaps[f"{tier}/{pname}"] = round(j, 3)
            print(f"{tier}/{pname}: {j:.3f}")
    results["overlap_vs_raw"] = overlaps

    # ── Sweep 2: exploration weight (sonata, medium) ──
    print("\n== exploration (sonata, medium100) ==")
    expl = {}
    for e in (0.1, 0.3, 0.9):
        r = HybridRanker(pq_path=MODELS / "sonata_pq.npz",
                         exploration=e)
        recs, ms = timed_recs(r, profiles["medium100"], k=20)
        expl[str(e)] = {"ms": round(ms, 1), "top": recs[:10]}
        print(f"exploration={e}: {ms:.1f} ms  top5={recs[:5]}")
    results["exploration_sonata_medium"] = expl

    # ── Sweep 3: context on/off, peak vs off-peak (sonata, medium) ──
    print("\n== context (sonata, medium100) ==")
    ctx = {}
    for label, ts in (("none", None), ("peak", peak_ts), ("offpeak", off_ts)):
        kw = dict(k=20)
        if ts is not None:
            kw.update(rhythm=rhythm, context_now=ts)
        recs, ms = timed_recs(rankers["sonata"], profiles["medium100"], **kw)
        ctx[label] = {"ms": round(ms, 1), "top": recs[:10]}
        print(f"context={label}: {ms:.1f} ms  top5={recs[:5]}")
    ctx["overlap_peak_vs_none"] = round(
        jaccard(ctx["peak"]["top"], ctx["none"]["top"]), 3)
    ctx["overlap_offpeak_vs_none"] = round(
        jaccard(ctx["offpeak"]["top"], ctx["none"]["top"]), 3)
    results["context_sonata_medium"] = ctx

    # ── Sweep 4: half-life (sonata, heavy) ──
    print("\n== half-life (sonata, heavy300) ==")
    hl = {}
    for h in (30.0, 90.0, 365.0):
        recs, ms = timed_recs(rankers["sonata"], profiles["heavy300"],
                             k=20, half_life_days=h)
        hl[str(int(h))] = {"ms": round(ms, 1), "top": recs[:10]}
        print(f"half_life={h}: {ms:.1f} ms  top5={recs[:5]}")
    results["halflife_sonata_heavy"] = hl

    # ── Sweep 5: exclude_played + k (sonata, light) ──
    print("\n== exclude_played / k (sonata, light30) ==")
    ep = {}
    for excl in (True, False):
        for k in (10, 20):
            recs, ms = timed_recs(rankers["sonata"], profiles["light30"],
                                  k=k, exclude_played=excl)
            played = set(profiles["light30"].item_counts)
            rep = sum(1 for x in recs if x in played)
            ep[f"excl{int(excl)}_k{k}"] = {
                "ms": round(ms, 1), "repeat_share": round(rep / len(recs), 3),
                "top": recs[:10]}
            print(f"excl={excl} k={k}: {ms:.1f} ms  repeat_share={rep}/{len(recs)}")
    results["exclude_played_sonata_light"] = ep

    # ── Latency stats per tier across all calls ──
    print("\n== latency summary (all calls in this sweep) ==")
    lat = {}
    for tier in TIERS:
        ms = [r["ms"] for r in results["runs"] if r["tier"] == tier]
        if ms:
            lat[tier] = {"n": len(ms), "p50": round(statistics.median(ms), 1),
                         "min": round(min(ms), 1), "max": round(max(ms), 1)}
            print(f"{tier}: n={len(ms)} p50={statistics.median(ms):.1f} "
                  f"min={min(ms):.1f} max={max(ms):.1f}")
    results["latency_summary"] = lat

    out = ROOT / "reports" / "sweep_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
