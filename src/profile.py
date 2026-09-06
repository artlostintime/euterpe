"""Local listening profile — on-device state for a single user."""

from __future__ import annotations

import time
from typing import Dict

# ------------------------------------------------------------------
# Signal types (Phase 11)
# ------------------------------------------------------------------

SIGNAL_WEIGHTS: Dict[str, float] = {
    "play": 1.0,
    "completion": 1.2,
    "replay": 1.0,
    "like": 2.0,
    "save": 1.5,
    "playlist_add": 1.5,
    "short_play": 0.3,
    "skip": 0.0,
}


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
        weighted_scores: Dict[int, float] | None = None,
        exploration: float = 0.3,
    ) -> None:
        self.item_counts: Dict[int, int] = dict(item_counts) if item_counts else {}
        self.last_seen_ts: Dict[int, float] = dict(last_seen_ts) if last_seen_ts else {}
        self.created_at: float = created_at if created_at is not None else time.time()
        self.weighted_scores: Dict[int, float] = dict(weighted_scores) if weighted_scores else {}
        self.exploration: float = max(0.1, min(0.9, float(exploration)))
        # Backward compat: derive weighted_scores from item_counts when absent
        if not self.weighted_scores and self.item_counts:
            self.weighted_scores = {k: float(v) for k, v in self.item_counts.items()}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def observe(self, item_id: int, ts: float | None = None, signal: str = "play") -> None:
        """Record an event.  *ts* defaults to now (unix seconds)."""
        if signal not in SIGNAL_WEIGHTS:
            raise ValueError(f"Unknown signal: {signal!r}")
        if ts is None:
            ts = time.time()
        self.item_counts[item_id] = self.item_counts.get(item_id, 0) + 1
        prev = self.last_seen_ts.get(item_id, 0.0)
        if ts > prev:
            self.last_seen_ts[item_id] = ts
        self.weighted_scores[item_id] = (
            self.weighted_scores.get(item_id, 0.0) + SIGNAL_WEIGHTS[signal]
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def decay_scores(self, half_life_days: float, now: float | None = None) -> Dict[int, float]:
        """Exponential-decay score per played item.

        score = weighted_score * 0.5 ** ((now - last_seen) / (half_life_days * 86400))

        Items never played get no entry.
        """
        if now is None:
            now = time.time()
        half_life_sec = half_life_days * 86400.0
        scores: Dict[int, float] = {}
        for item_id, wscore in self.weighted_scores.items():
            last = self.last_seen_ts.get(item_id, 0.0)
            elapsed = max(0.0, now - last)
            scores[item_id] = wscore * (0.5 ** (elapsed / half_life_sec))
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
            "weighted_scores": {str(k): v for k, v in self.weighted_scores.items()},
            "exploration": self.exploration,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LocalProfile:
        """Reconstruct from a :pymeth:`to_dict` output."""
        ic = {int(k): v for k, v in d.get("item_counts", {}).items()}
        ls = {int(k): v for k, v in d.get("last_seen_ts", {}).items()}
        ws = {int(k): v for k, v in d.get("weighted_scores", {}).items()}
        return cls(
            item_counts=ic,
            last_seen_ts=ls,
            created_at=d.get("created_at"),
            weighted_scores=ws,
            exploration=d.get("exploration", 0.3),
        )
