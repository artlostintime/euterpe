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


def test_context_pure_repeat_invariant() -> None:
    """Pure-repeat ranker: uniform rhythm scales the only score component
    uniformly, which is argsort-invariant — ranking unchanged."""
    _header("test_context_pure_repeat_invariant")
    rng = np.random.default_rng(42)
    n_items, dim = 50, 8
    emb = rng.standard_normal((n_items, dim)).astype(np.float32)
    tmp = Path(__file__).resolve().parent / "_test_ctx_emb.npy"
    np.save(tmp, emb)

    try:
        ranker = HybridRanker(embeddings_path=tmp, w_repeat=1.0, w_discovery=0.0)
        profile = LocalProfile()
        now = time.time()
        for i in range(10):
            profile.observe(i, now - i * 86400)

        rhythm = ListenRhythm()  # default = uniform
        recs_base = ranker.recommend(profile, k=15, exclude_played=False)
        recs_ctx = ranker.recommend(profile, k=15, exclude_played=False,
                                     rhythm=rhythm, context_now=now)
        assert recs_base == recs_ctx, (
            f"pure-repeat ranking changed under uniform scale: {recs_base} vs {recs_ctx}"
        )
        print("PASS")
    finally:
        tmp.unlink(missing_ok=True)


def test_context_flips_hybrid_order() -> None:
    """Context is live: peak hour boosts the repeat component enough to
    reorder a repeat-heavy item above a discovery-heavy one (and vice
    versa off-peak). Regression test for the uniform-scale no-op bug.

    Geometry (taste center points +x after exploration blend):"""
    _header("test_context_flips_hybrid_order")
    """
      C(0): 10 plays, emb [ 1, 0]     -> repeat 1.0, disc 1.0
      A(1):  4 plays, emb [-1, 0]     -> repeat 0.4, disc 0.0 (min)
      B(2):  0 plays, emb [-0.2, .98] -> repeat 0.0, disc 0.4
      D(3):  0 plays, emb [-0.8, .6]  -> disc 0.1
      E(4):  0 plays, emb [-0.5,-.87] -> disc 0.25
    Off-peak (0.9): hybrid = 0.45*rep + 0.5*disc -> B(0.20) > A(0.18).
    Peak    (1.1): hybrid = 0.55*rep + 0.5*disc -> A(0.22) > B(0.20).
    """
    emb = np.array([
        [1.0, 0.0],
        [-1.0, 0.0],
        [-0.2, 0.9799],
        [-0.8, 0.6],
        [-0.5, -0.866025],
    ], dtype=np.float32)
    tmp = Path(__file__).resolve().parent / "_test_ctx_flip.npy"
    np.save(tmp, emb)

    # All listens Monday 22:00 — rhythm: hour 22 prob 1, Monday prob 1.
    base_dt = datetime.datetime(2024, 1, 15, 22, 0)  # Monday
    ts = base_dt.timestamp()
    peak_ts = ts
    off_ts = (base_dt + datetime.timedelta(hours=16)).timestamp()  # Tue 14:00

    try:
        ranker = HybridRanker(embeddings_path=tmp,
                              w_repeat=0.5, w_discovery=0.5)
        profile = LocalProfile()
        for _ in range(10):
            profile.observe(0, ts)
        for _ in range(4):
            profile.observe(1, ts)

        rhythm = ListenRhythm()
        rhythm.learn([ts] * 14)
        assert abs(rhythm.context_weight(peak_ts) - 1.1) < 1e-9
        assert abs(rhythm.context_weight(off_ts) - 0.9) < 1e-9

        recs_peak = ranker.recommend(profile, k=5, exclude_played=False,
                                      rhythm=rhythm, context_now=peak_ts)
        recs_off = ranker.recommend(profile, k=5, exclude_played=False,
                                     rhythm=rhythm, context_now=off_ts)
        ia_p, ib_p = recs_peak.index(1), recs_peak.index(2)
        ia_o, ib_o = recs_off.index(1), recs_off.index(2)
        assert ia_p < ib_p, f"peak: repeat-heavy A should outrank B, got {recs_peak}"
        assert ia_o > ib_o, f"off-peak: discovery-heavy B should outrank A, got {recs_off}"
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
    test_context_pure_repeat_invariant,
    test_context_flips_hybrid_order,
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
