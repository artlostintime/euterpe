"""Tests for Phase 16 contribution builder — pytest-free, stdlib only."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.contribution import ContributionBuilder
from src.profile import LocalProfile


def _header(name: str) -> None:
    print(f"--- {name} ---")


def test_consent_gate() -> None:
    """build() with consent=False raises ValueError; with consent=True works."""
    _header("test_consent_gate")
    p = LocalProfile()
    p.observe(1, time.time())
    b_no = ContributionBuilder(p, consent=False)
    try:
        b_no.build()
        assert False, "should have raised ValueError"
    except ValueError:
        pass
    b_yes = ContributionBuilder(p, consent=True)
    result = b_yes.build()
    assert result["schema"] == "contribution.v1"
    print("PASS")


def test_minimization() -> None:
    """Only items with count >= min_count appear in the payload."""
    _header("test_minimization")
    p = LocalProfile()
    now = time.time()
    # counts 1, 2, 3, 10, 50 — only 10 and 50 survive min_count=5
    p.item_counts = {1: 1, 2: 2, 3: 3, 10: 10, 50: 50}
    p.last_seen_ts = {k: now for k in p.item_counts}
    b = ContributionBuilder(p, consent=True)
    payload = b.build(min_count=5)
    ids = {item["item_id"] for item in payload["items"]}
    counts = {item["item_id"]: item["count"] for item in payload["items"]}
    assert ids == {10, 50}, f"unexpected items: {ids}"
    assert payload["n_items"] == 2, f"n_items: {payload['n_items']}"
    assert counts[50] == 50 and counts[10] == 10
    print("PASS")


def test_max_items_cap() -> None:
    """300 eligible items, max_items=200 → exactly 200, highest counts."""
    _header("test_max_items_cap")
    p = LocalProfile()
    now = time.time()
    # 300 items with counts 1..300
    for i in range(1, 301):
        p.item_counts[i] = i
        p.last_seen_ts[i] = now
    b = ContributionBuilder(p, consent=True)
    payload = b.build(max_items=200, min_count=1)
    assert payload["n_items"] == 200, f"n_items: {payload['n_items']}"
    # Verify they're the 200 highest-count items (counts 101..300)
    counts_sorted = sorted(item["count"] for item in payload["items"])
    assert counts_sorted[0] == 101, f"min count: {counts_sorted[0]}"
    assert counts_sorted[-1] == 300, f"max count: {counts_sorted[-1]}"
    print("PASS")


def test_coarse_ts() -> None:
    """coarse_ts replaces exact timestamps with YYYY-MM month buckets."""
    _header("test_coarse_ts")
    p = LocalProfile()
    # item 1: last seen March 15 2024 14:30 → should become "2024-03"
    import datetime
    ts_exact = datetime.datetime(2024, 3, 15, 14, 30).timestamp()
    p.item_counts[1] = 10
    p.last_seen_ts[1] = ts_exact
    b = ContributionBuilder(p, consent=True)
    payload = b.build(coarse_ts=True, min_count=1)
    item = payload["items"][0]
    assert item["last_seen_month"] == "2024-03", f"got {item['last_seen_month']}"
    # coarse_ts=False omits the field
    payload2 = b.build(coarse_ts=False, min_count=1)
    assert "last_seen_month" not in payload2["items"][0]
    print("PASS")


ALL_TESTS = [
    test_consent_gate,
    test_minimization,
    test_max_items_cap,
    test_coarse_ts,
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
