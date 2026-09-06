"""Tests for the recommendation runtime — pytest-free, stdlib only."""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

import numpy as np

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.profile import LocalProfile
from src.ranker import HybridRanker, _min_max, _pq_decode


def _header(name: str) -> None:
    print(f"--- {name} ---")


def test_profile_observe() -> None:
    _header("test_profile_observe")
    p = LocalProfile()
    p.observe(1, 1000.0)
    p.observe(1, 2000.0)
    p.observe(2, 1500.0)
    assert p.item_counts == {1: 2, 2: 1}, f"counts: {p.item_counts}"
    assert p.last_seen_ts[1] == 2000.0
    assert p.last_seen_ts[2] == 1500.0
    print("PASS")


def test_decay_math() -> None:
    _header("test_decay_math")
    p = LocalProfile()
    now = 1_700_000_000.0  # realistic unix timestamp (Nov 2023)
    # item 1: played 3 times, seen right now
    p.observe(1, now)
    p.observe(1, now)
    p.observe(1, now)
    # item 2: played 1 time, seen 90 days ago (one half-life)
    p.observe(2, now - 90 * 86400)
    # item 3: played 2 times, seen 180 days ago (two half-lives)
    p.observe(3, now - 180 * 86400)
    p.observe(3, now - 180 * 86400)

    scores = p.decay_scores(half_life_days=90, now=now)
    # item1: 3 * 0.5^0 = 3.0
    assert abs(scores[1] - 3.0) < 1e-9, f"item1 score: {scores[1]}"
    # item2: 1 * 0.5^1 = 0.5
    assert abs(scores[2] - 0.5) < 1e-9, f"item2 score: {scores[2]}"
    # item3: 2 * 0.5^2 = 0.5
    assert abs(scores[3] - 0.5) < 1e-9, f"item3 score: {scores[3]}"
    print("PASS")


def test_json_roundtrip() -> None:
    _header("test_json_roundtrip")
    p = LocalProfile()
    p.observe(10, 100.0)
    p.observe(20, 200.0)
    d = p.to_dict()
    p2 = LocalProfile.from_dict(d)
    assert p2.item_counts == p.item_counts
    assert p2.last_seen_ts == p.last_seen_ts
    assert p2.created_at == p.created_at
    print("PASS")


def test_recommend_basic() -> None:
    _header("test_recommend_basic")
    rng = np.random.default_rng(0)
    n_items, dim = 100, 8
    emb = rng.standard_normal((n_items, dim)).astype(np.float32)
    # save to temp file
    tmp = Path(__file__).resolve().parent / "_test_emb.npy"
    np.save(tmp, emb)

    try:
        ranker = HybridRanker(embeddings_path=tmp)
        assert ranker.has_embeddings()

        profile = LocalProfile()
        now = time.time()
        for i in range(10):
            profile.observe(i, now - i * 86400)

        k = 15
        recs = ranker.recommend(profile, k=k, exclude_played=True)
        assert len(recs) == k, f"expected {k} recs, got {len(recs)}"
        # no played items in results
        played = set(profile.item_counts.keys())
        assert not any(r in played for r in recs), f"played items leaked: {[r for r in recs if r in played]}"
        # all ids are valid
        assert all(0 <= r < n_items for r in recs)
        print("PASS")
    finally:
        tmp.unlink(missing_ok=True)


def test_exclude_played_false() -> None:
    _header("test_exclude_played_false")
    rng = np.random.default_rng(1)
    n_items, dim = 50, 8
    emb = rng.standard_normal((n_items, dim)).astype(np.float32)
    tmp = Path(__file__).resolve().parent / "_test_emb2.npy"
    np.save(tmp, emb)

    try:
        ranker = HybridRanker(embeddings_path=tmp)
        profile = LocalProfile()
        now = time.time()
        # play item 0 with very high recency so it dominates
        for _ in range(20):
            profile.observe(0, now)

        recs = ranker.recommend(profile, k=5, exclude_played=False)
        assert 0 in recs, "played item 0 should appear when exclude_played=False"
        print("PASS")
    finally:
        tmp.unlink(missing_ok=True)


def test_pq_decode_roundtrip() -> None:
    _header("test_pq_decode_roundtrip")
    rng = np.random.default_rng(99)
    n_items, n_sub, k, sub_dim = 20, 2, 4, 3
    # build centroids: random float32
    centroids = rng.standard_normal((n_sub, k, sub_dim)).astype(np.float32)
    # build codes: random uint8 in [0, k)
    codes = rng.integers(0, k, size=(n_items, n_sub), dtype=np.uint8)
    # reconstruct
    reconstructed = _pq_decode(codes, centroids)
    assert reconstructed.shape == (n_items, n_sub * sub_dim)
    # verify each row matches the expected concatenation
    for i in range(n_items):
        expected = np.concatenate([centroids[s, codes[i, s]] for s in range(n_sub)])
        assert np.allclose(reconstructed[i], expected, atol=1e-7), f"row {i} mismatch"
    # roundtrip: save and load
    tmp = Path(__file__).resolve().parent / "_test_pq.npz"
    np.savez(tmp, codes=codes, centroids=centroids, item_ids=np.arange(n_items))
    try:
        ranker = HybridRanker(pq_path=tmp)
        assert ranker.has_embeddings()
        assert ranker._embeddings.shape == (n_items, n_sub * sub_dim)
        print("PASS")
    finally:
        tmp.unlink(missing_ok=True)


def test_min_max() -> None:
    _header("test_min_max")
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    normed = _min_max(a)
    assert abs(normed[0]) < 1e-9 and abs(normed[4] - 1.0) < 1e-9
    # constant array
    c = np.array([7.0, 7.0, 7.0])
    assert all(x == 0.0 for x in _min_max(c))
    print("PASS")


def test_recommend_ranking_order() -> None:
    _header("test_recommend_ranking_order")
    """Repeat-dominant profile should rank played items high when not excluded."""
    rng = np.random.default_rng(7)
    n_items, dim = 30, 8
    emb = rng.standard_normal((n_items, dim)).astype(np.float32)
    tmp = Path(__file__).resolve().parent / "_test_emb3.npy"
    np.save(tmp, emb)

    try:
        ranker = HybridRanker(embeddings_path=tmp, w_repeat=1.0, w_discovery=0.0)
        profile = LocalProfile()
        now = time.time()
        # play item 0 many times, very recently
        for _ in range(50):
            profile.observe(0, now)

        recs = ranker.recommend(profile, k=5, exclude_played=False)
        assert recs[0] == 0, f"item 0 should be #1 with w_repeat=1.0, got {recs}"
        print("PASS")
    finally:
        tmp.unlink(missing_ok=True)


def test_build_real_profile() -> None:
    """_build_real_profile picks IDs that exist in a fake top_items.json."""
    _header("test_build_real_profile")
    import json
    from src.__main__ import _build_real_profile

    # Build a fake top_items.json with 200 entries
    fake_items = [{"v1_item_id": i, "recording_mbid": f"{i:032x}", "count": 100000 + i} for i in range(200)]
    tmp_dir = Path(__file__).resolve().parent / "_test_real_profile_tmp"
    tmp_dir.mkdir(exist_ok=True)
    fake_path = tmp_dir / "top_items.json"
    try:
        fake_path.write_text(json.dumps(fake_items))
        profile = _build_real_profile(fake_path, n_items=30, span_days=90)
        # Profile should have items picked from the first 200
        assert len(profile.item_counts) > 0, "profile is empty"
        assert len(profile.item_counts) <= 30, f"too many items: {len(profile.item_counts)}"
        # Every profile item_id must be in [0, 199] — within the fake top_items
        for item_id in profile.item_counts:
            assert 0 <= item_id < 200, f"item_id {item_id} out of range"
        # Deterministic: same call → same IDs
        profile2 = _build_real_profile(fake_path, n_items=30, span_days=90)
        assert set(profile.item_counts.keys()) == set(profile2.item_counts.keys()), "non-deterministic"
        print("PASS")
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


ALL_TESTS = [
    test_profile_observe,
    test_decay_math,
    test_json_roundtrip,
    test_recommend_basic,
    test_exclude_played_false,
    test_pq_decode_roundtrip,
    test_min_max,
    test_recommend_ranking_order,
    test_build_real_profile,
]

if __name__ == "__main__":
    passed = 0
    failed = 0
    for t in ALL_TESTS:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}")
            failed += 1
    print(f"\n{'='*40}")
    print(f"Results: {passed} PASS, {failed} FAIL out of {len(ALL_TESTS)}")
    if failed:
        sys.exit(1)
