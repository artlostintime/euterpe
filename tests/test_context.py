"""Tests for context signals — pytest-free, stdlib only."""

from __future__ import annotations

import datetime
import sys
import time
from pathlib import Path

import numpy as np

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.context import ListenRhythm
from src.profile import LocalProfile
from src.ranker import HybridRanker


def _header(name: str) -> None:
    print(f"--- {name} ---")


def _ts(hour: int = 22, day: int = 15, month: int = 1) -> float:
    """Create a local-time unix timestamp for testing."""
    return datetime.datetime(2024, month, day, hour, 0).timestamp()


def test_learn_correct_bucket() -> None:
    """All listens at hour 22 → hour 22 has probability 1.0."""
    _header("test_learn_correct_bucket")
    rhythm = ListenRhythm()
    # All 10 listens at 22:00 on Jan 15 2024 (Monday, weekday=0)
    timestamps = [_ts(hour=22, day=15) for _ in range(10)]
    rhythm.learn(timestamps)
    assert abs(rhythm.hour_probs[22] - 1.0) < 1e-9, f"hour 22 prob: {rhythm.hour_probs[22]}"
    assert abs(sum(rhythm.hour_probs) - 1.0) < 1e-9, "hour probs don't sum to 1"
    # Day probs: all on Monday (weekday=0)
    assert abs(rhythm.day_probs[0] - 1.0) < 1e-9, f"Monday prob: {rhythm.day_probs[0]}"
    assert abs(sum(rhythm.day_probs) - 1.0) < 1e-9, "day probs don't sum to 1"
    print("PASS")


def test_boost_cap() -> None:
    """Extreme rhythm (all at one hour) yields factor within [0.9, 1.1]."""
    _header("test_boost_cap")
    rhythm = ListenRhythm()
    # 100 listens all at hour 22 on Monday
    timestamps = [_ts(hour=22, day=15) for _ in range(100)]
    rhythm.learn(timestamps)
    # Peak hour
    w_peak = rhythm.context_weight(_ts(hour=22, day=15))
    assert 0.9 <= w_peak <= 1.1, f"peak weight {w_peak} out of [0.9, 1.1]"
    # Different hour (5 AM)
    w_low = rhythm.context_weight(_ts(hour=5, day=15))
    assert 0.9 <= w_low <= 1.1, f"low weight {w_low} out of [0.9, 1.1]"
    assert w_peak > w_low, f"peak {w_peak} should exceed low {w_low}"
    print("PASS")


def test_roundtrip_dict() -> None:
    """to_dict/from_dict preserves hour and day probabilities."""
    _header("test_roundtrip_dict")
    rhythm = ListenRhythm()
    timestamps = [_ts(hour=22, day=15) for _ in range(10)] + [_ts(hour=8, day=16) for _ in range(5)]
    rhythm.learn(timestamps)
    d = rhythm.to_dict()
    rhythm2 = ListenRhythm.from_dict(d)
    assert np.allclose(rhythm.hour_probs, rhythm2.hour_probs), "hour_probs mismatch"
    assert np.allclose(rhythm.day_probs, rhythm2.day_probs), "day_probs mismatch"
    print("PASS")


def test_context_uniform_no_ranking_change() -> None:
    """Uniform rhythm: context multiplication doesn't change ranking order."""
    _header("test_context_uniform_no_ranking_change")
    rng = np.random.default_rng(42)
    n_items, dim = 50, 8
    emb = rng.standard_normal((n_items, dim)).astype(np.float32)
    tmp = Path(__file__).resolve().parent / "_test_ctx_emb.npy"
    np.save(tmp, emb)

    try:
        ranker = HybridRanker(embeddings_path=tmp)
        profile = LocalProfile()
        now = time.time()
        for i in range(10):
            profile.observe(i, now - i * 86400)

        rhythm = ListenRhythm()  # default = uniform
        recs_base = ranker.recommend(profile, k=15)
        recs_ctx = ranker.recommend(profile, k=15, rhythm=rhythm, context_now=now)
        assert recs_base == recs_ctx, (
            f"ranking changed with uniform rhythm: {recs_base} vs {recs_ctx}"
        )
        print("PASS")
    finally:
        tmp.unlink(missing_ok=True)


def test_context_none_bitidentical() -> None:
    """context_now=None is bit-identical to the old path."""
    _header("test_context_none_bitidentical")
    rng = np.random.default_rng(7)
    n_items, dim = 40, 8
    emb = rng.standard_normal((n_items, dim)).astype(np.float32)
    tmp = Path(__file__).resolve().parent / "_test_ctx_emb2.npy"
    np.save(tmp, emb)

    try:
        ranker = HybridRanker(embeddings_path=tmp)
        profile = LocalProfile()
        now = time.time()
        for i in range(8):
            profile.observe(i, now - i * 86400)

        # No rhythm at all
        recs_base = ranker.recommend(profile, k=10)
        # With rhythm but context_now=None → no multiplication
        rhythm = ListenRhythm()
        recs_none = ranker.recommend(profile, k=10, rhythm=rhythm, context_now=None)
        assert recs_base == recs_none, (
            f"not bit-identical: {recs_base} vs {recs_none}"
        )
        print("PASS")
    finally:
        tmp.unlink(missing_ok=True)


ALL_TESTS = [
    test_learn_correct_bucket,
    test_boost_cap,
    test_roundtrip_dict,
    test_context_uniform_no_ranking_change,
    test_context_none_bitidentical,
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
