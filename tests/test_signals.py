"""Tests for Phase 11 local signal types — pytest-free, stdlib only."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.profile import LocalProfile, SIGNAL_WEIGHTS


def _header(name: str) -> None:
    print(f"--- {name} ---")


def test_default_signal_backward_compat() -> None:
    """observe(item, ts) with no signal arg → decay_scores identical to old-style."""
    _header("test_default_signal_backward_compat")
    now = 1_700_000_000.0

    # Old-style: load from JSON without weighted_scores (triggers derivation)
    old_json = {
        "item_counts": {"1": 3, "2": 1},
        "last_seen_ts": {"1": now, "2": now - 90 * 86400},
        "created_at": 0.0,
    }
    old = LocalProfile.from_dict(old_json)

    # New-style: observe with default signal
    new = LocalProfile()
    for _ in range(3):
        new.observe(1, now)
    new.observe(2, now - 90 * 86400)

    old_scores = old.decay_scores(half_life_days=90, now=now)
    new_scores = new.decay_scores(half_life_days=90, now=now)
    assert set(old_scores.keys()) == set(new_scores.keys()), f"keys differ: {old_scores} vs {new_scores}"
    for k in old_scores:
        assert abs(old_scores[k] - new_scores[k]) < 1e-9, f"item {k}: {old_scores[k]} vs {new_scores[k]}"
    print("PASS")


def test_weight_math() -> None:
    """like (2.0) outranks play (1.0); skip contributes 0."""
    _header("test_weight_math")
    now = 1_700_000_000.0

    p = LocalProfile()
    # item A: 3 likes (weight 2.0 each → 6.0)
    for _ in range(3):
        p.observe(1, now, signal="like")
    # item B: 3 plays (weight 1.0 each → 3.0)
    for _ in range(3):
        p.observe(2, now, signal="play")
    # item C: 5 skips (weight 0.0 each → 0.0)
    for _ in range(5):
        p.observe(3, now, signal="skip")

    scores = p.decay_scores(half_life_days=90, now=now)
    assert scores[1] > scores[2], f"like ({scores[1]}) should outrank play ({scores[2]})"
    assert scores[3] == 0.0, f"skip should be 0.0, got {scores[3]}"
    print("PASS")


def test_unknown_signal_raises() -> None:
    """observe with unknown signal raises ValueError."""
    _header("test_unknown_signal_raises")
    p = LocalProfile()
    try:
        p.observe(1, 1000.0, signal="invalid")
        assert False, "should have raised ValueError"
    except ValueError:
        pass
    print("PASS")


def test_exploration_roundtrip() -> None:
    """exploration roundtrip; missing field → 0.3; clamp works."""
    _header("test_exploration_roundtrip")
    p = LocalProfile()
    p.exploration = 0.7
    d = p.to_dict()
    p2 = LocalProfile.from_dict(d)
    assert abs(p2.exploration - 0.7) < 1e-9, f"roundtrip: {p2.exploration}"

    # Missing field in old JSON → 0.3
    old_json = {"item_counts": {"1": 5}, "last_seen_ts": {"1": 1000.0}, "created_at": 999.0}
    p3 = LocalProfile.from_dict(old_json)
    assert abs(p3.exploration - 0.3) < 1e-9, f"default: {p3.exploration}"

    # Clamp
    p4 = LocalProfile(exploration=5.0)
    assert abs(p4.exploration - 0.9) < 1e-9, f"clamp high: {p4.exploration}"
    p5 = LocalProfile(exploration=-1.0)
    assert abs(p5.exploration - 0.1) < 1e-9, f"clamp low: {p5.exploration}"
    print("PASS")


def test_weighted_json_roundtrip() -> None:
    """Weighted profile JSON roundtrip preserves weights."""
    _header("test_weighted_json_roundtrip")
    now = 1_700_000_000.0
    p = LocalProfile()
    p.observe(1, now, signal="like")
    p.observe(1, now + 1, signal="save")
    p.observe(2, now, signal="play")
    p.observe(2, now + 1, signal="short_play")
    p.exploration = 0.6

    d = p.to_dict()
    p2 = LocalProfile.from_dict(d)

    assert set(p.weighted_scores.keys()) == set(p2.weighted_scores.keys()), "keys differ"
    for k in p.weighted_scores:
        assert abs(p.weighted_scores[k] - p2.weighted_scores[k]) < 1e-9, (
            f"item {k}: {p.weighted_scores[k]} vs {p2.weighted_scores[k]}"
        )
    assert abs(p2.exploration - 0.6) < 1e-9, f"exploration: {p2.exploration}"
    print("PASS")


ALL_TESTS = [
    test_default_signal_backward_compat,
    test_weight_math,
    test_unknown_signal_raises,
    test_exploration_roundtrip,
    test_weighted_json_roundtrip,
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
