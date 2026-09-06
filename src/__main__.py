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

from .profile import LocalProfile
from .ranker import HybridRanker


def _build_synthetic_profile(n_items: int = 50, span_days: int = 90) -> LocalProfile:
    """Create a fake profile with *n_items* items spread over *span_days*."""
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

    # -- profile --
    profile = _build_synthetic_profile()
    print(f"Profile: {len(profile.item_counts)} items played")

    # -- embeddings --
    emb_path, pq_path = HybridRanker.discover_embeddings()
    ranker = HybridRanker(embeddings_path=emb_path, pq_path=pq_path)

    if not ranker.has_embeddings():
        print(
            "No embeddings or PQ codes found in search paths. "
            "Skipping recommend() demo (tests cover this with synthetic data).\n"
            "Searched:"
        )
        for d in HybridRanker.SEARCH_DIRS:
            print(f"  {d}")
        print("\nTo run the full demo, place a .npy (2-D float) or .npz "
              "(keys: codes, centroids) under one of the above paths.")
        return

    src = emb_path or pq_path
    print(f"Loaded embeddings from: {src}")

    recs = ranker.recommend(profile, k=10)
    print(f"\nTop-10 recs (ids): {recs}")

    # sanity: no played items should appear in discovery-dominant recs
    played = set(profile.item_counts.keys())
    overlap = [r for r in recs if r in played]
    if overlap:
        print(f"  NOTE: {overlap} are played — repeat signal dominated (expected with w_repeat=0.7)")
    else:
        print("  All recs are unplayed items (discovery working)")

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


def main() -> None:
    ap = argparse.ArgumentParser(description="Recommendation runtime demo")
    ap.add_argument("--demo", action="store_true", help="Run synthetic self-test")
    ap.add_argument("--profile", type=str, help="Path to profile JSON")
    ap.add_argument("--k", type=int, default=20, help="Number of recommendations")
    args = ap.parse_args()

    if args.demo:
        _run_demo()
    elif args.profile:
        _run_from_profile(args.profile, args.k)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
