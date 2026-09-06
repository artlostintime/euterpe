"""Local listening profile — on-device state for a single user."""

from __future__ import annotations

import time
from typing import Dict


class LocalProfile:
    """Compact on-device listening history.

    Stores per-item play counts and last-seen timestamps.
    Supports exponential-decay scoring and JSON round-trip.
    """

    def __init__(
        self,
        item_counts: Dict[int, int] | None = None,
        last_seen_ts: Dict[int, float] | None = None,
        created_at: float | None = None,
    ) -> None:
        self.item_counts: Dict[int, int] = dict(item_counts) if item_counts else {}
        self.last_seen_ts: Dict[int, float] = dict(last_seen_ts) if last_seen_ts else {}
        self.created_at: float = created_at if created_at is not None else time.time()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def observe(self, item_id: int, ts: float | None = None) -> None:
        """Record a play event.  *ts* defaults to now (unix seconds)."""
        if ts is None:
            ts = time.time()
        self.item_counts[item_id] = self.item_counts.get(item_id, 0) + 1
        prev = self.last_seen_ts.get(item_id, 0.0)
        if ts > prev:
            self.last_seen_ts[item_id] = ts

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def decay_scores(self, half_life_days: float, now: float | None = None) -> Dict[int, float]:
        """Exponential-decay score per played item.

        score = count * 0.5 ** ((now - last_seen) / (half_life_days * 86400))

        Items never played get no entry.
        """
        if now is None:
            now = time.time()
        half_life_sec = half_life_days * 86400.0
        scores: Dict[int, float] = {}
        for item_id, count in self.item_counts.items():
            last = self.last_seen_ts.get(item_id, 0.0)
            elapsed = max(0.0, now - last)
            scores[item_id] = count * (0.5 ** (elapsed / half_life_sec))
        return scores

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-serialisable dict."""
        return {
            "item_counts": {str(k): v for k, v in self.item_counts.items()},
            "last_seen_ts": {str(k): v for k, v in self.last_seen_ts.items()},
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LocalProfile:
        """Reconstruct from a :pymeth:`to_dict` output."""
        ic = {int(k): v for k, v in d.get("item_counts", {}).items()}
        ls = {int(k): v for k, v in d.get("last_seen_ts", {}).items()}
        return cls(item_counts=ic, last_seen_ts=ls, created_at=d.get("created_at"))
