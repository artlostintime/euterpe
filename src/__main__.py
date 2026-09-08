"""CLI entry-point for the recommendation runtime skeleton.

Usage::

    python -m src --demo          # synthetic self-test
    python -m src --profile p.json --k 20   # score a real profile
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from .contribution import ContributionBuilder
from .profile import LocalProfile
from .ranker import HybridRanker


def _build_real_profile(top_items_path: Path, n_items: int = 30, span_days: int = 90) -> LocalProfile:
    """Load top_items.json and pick n_items deterministically (every 6th from first 200).

    Assigns synthetic timestamps spread over span_days with varying play counts.
    Item IDs are v1_item_id values — guaranteed to exist in the embedding matrix.
    """
    data = json.loads(top_items_path.read_text())
    # Take first 200 entries, pick every 6th → 33 items, take first n_items
    pool = data[:200]
    selected = pool[::6][:n_items]

    rng = random.Random(42)
    now = time.time()
    p = LocalProfile()
    for entry in selected:
        item_id = entry["v1_item_id"]
        ts = now - rng.uniform(0, span_days * 86400)
        # play count 1-10, higher for more popular items
        base_plays = max(1, min(10, entry["count"] // 50000))
        plays = rng.randint(1, base_plays)
        for _ in range(plays):
            p.observe(item_id, ts + rng.uniform(0, 86400))
    return p


def _find_top_items() -> Path | None:
    """Locate top_items.json in any SEARCH_DIR."""
    for d in HybridRanker.search_dirs():
        p = d / "top_items.json"
        if p.exists():
            return p
    return None


def _build_synthetic_profile(n_items: int = 50, span_days: int = 90) -> LocalProfile:
    """Fallback: create a fake profile with *n_items* items that won't match any real embeddings."""
    rng = random.Random(42)
    now = time.time()
    p = LocalProfile()
    for i in range(n_items):
        item_id = 1000 + i
        ts = now - rng.uniform(0, span_days * 86400)
        for _ in range(rng.randint(1, 10)):
            p.observe(item_id, ts + rng.uniform(0, 86400))
    return p


def _run_demo() -> None:
    print("=== HybridRanker self-test ===\n")

    # -- profile: prefer real data, fall back to synthetic --
    top_items = _find_top_items()
    if top_items is not None:
        profile = _build_real_profile(top_items)
        print(f"Profile: {len(profile.item_counts)} items (from {top_items.name})")
    else:
        profile = _build_synthetic_profile()
        print(f"Profile: {len(profile.item_counts)} items (synthetic fallback)")

    # -- embeddings --
    emb_path, pq_path = HybridRanker.discover_embeddings()
    ranker = HybridRanker(
        embeddings_path=emb_path,
        pq_path=pq_path,
        exploration=profile.exploration,
    )

    if not ranker.has_embeddings():
        print(
            "No embeddings or PQ codes found in search paths. "
            "Skipping recommend() demo (tests cover this with synthetic data).\n"
            "Searched:"
        )
        for d in HybridRanker.search_dirs():
            print(f"  {d}")
        print("\nTo run the full demo, place a .npy (2-D float) or .npz "
              "(keys: codes, centroids) under one of the above paths.")
        return

    src = emb_path or pq_path
    # Load once to show shape (ranker already holds it)
    ranker._load()
    if ranker._embeddings is not None:
        shape = ranker._embeddings.shape
    else:
        n_sub, _, sub_dim = ranker._centroids.shape
        shape = (ranker._codes.shape[0], n_sub * sub_dim)
    print(f"Embeddings: {src}  shape={shape[0]}×{shape[1]}")

    recs = ranker.recommend(profile, k=10)
    print(f"\nTop-10 recs (v1_item_id): {recs}")

    # -- repeat vs discovery breakdown --
    played = set(profile.item_counts.keys())
    repeat_count = sum(1 for r in recs if r in played)
    discovery_count = len(recs) - repeat_count
    print(f"  Repeat (played): {repeat_count}  |  Discovery (unplayed): {discovery_count}")

    print("\nSelf-test PASSED.")


def _run_from_profile(path: str, k: int) -> None:
    raw = json.loads(Path(path).read_text())
    profile = LocalProfile.from_dict(raw)
    print(f"Loaded profile: {len(profile.item_counts)} items")

    emb_path, pq_path = HybridRanker.discover_embeddings()
    ranker = HybridRanker(embeddings_path=emb_path, pq_path=pq_path)

    if not ranker.has_embeddings():
        print("No embeddings found — cannot produce recommendations.")
        sys.exit(1)

    recs = ranker.recommend(profile, k=k)
    print(f"Top-{k} recs: {recs}")


def _run_contribute(profile_path: str | None) -> None:
    """Build and print a privacy-filtered contribution payload."""
    if profile_path:
        raw = json.loads(Path(profile_path).read_text())
        profile = LocalProfile.from_dict(raw)
        print(f"Loaded profile: {len(profile.item_counts)} items")
    else:
        top_items = _find_top_items()
        if top_items is not None:
            profile = _build_real_profile(top_items)
            print(f"Profile: {len(profile.item_counts)} items (from {top_items.name})")
        else:
            profile = _build_synthetic_profile()
            print(f"Profile: {len(profile.item_counts)} items (synthetic fallback)")

    builder = ContributionBuilder(profile, consent=True)
    payload = builder.build()
    print(f"\nContribution payload: {payload['n_items']} items "
          f"(schema={payload['schema']})")
    print(json.dumps(payload, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Recommendation runtime demo")
    ap.add_argument("--demo", action="store_true", help="Run synthetic self-test")
    ap.add_argument("--profile", type=str, help="Path to profile JSON")
    ap.add_argument("--k", type=int, default=20, help="Number of recommendations")
    ap.add_argument("--contribute", action="store_true",
                    help="Build a privacy-filtered contribution payload")
    args = ap.parse_args()

    if args.demo:
        _run_demo()
    elif args.contribute:
        _run_contribute(args.profile)
    elif args.profile:
        _run_from_profile(args.profile, args.k)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
