"""Context signals — per-bucket listen-rate histograms for time-of-day / day-of-week weighting."""

from __future__ import annotations

import datetime
from typing import Sequence

import numpy as np


class ListenRhythm:
    """Hour-of-day (24) and day-of-week (7) listen-rate histograms.

    Normalised to probabilities.  Used to derive a context-weight
    multiplier in [0.9, 1.1] for the hybrid ranker.
    """

    N_HOURS = 24
    N_DAYS = 7
    _W_HOUR = 0.7
    _W_DAY = 0.3

    def __init__(
        self,
        hour_probs: np.ndarray | None = None,
        day_probs: np.ndarray | None = None,
    ) -> None:
        self.hour_probs: np.ndarray = (
            np.asarray(hour_probs, dtype=np.float64)
            if hour_probs is not None
            else np.full(self.N_HOURS, 1.0 / self.N_HOURS)
        )
        self.day_probs: np.ndarray = (
            np.asarray(day_probs, dtype=np.float64)
            if day_probs is not None
            else np.full(self.N_DAYS, 1.0 / self.N_DAYS)
        )

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn(self, timestamps: Sequence[float]) -> None:
        """Build histograms from unix timestamps, normalise to probabilities.

        Falls back to uniform if no timestamps are provided.
        """
        hour_counts = np.zeros(self.N_HOURS, dtype=np.float64)
        day_counts = np.zeros(self.N_DAYS, dtype=np.float64)
        for ts in timestamps:
            dt = datetime.datetime.fromtimestamp(ts)
            hour_counts[dt.hour] += 1
            day_counts[dt.weekday()] += 1
        h_sum = hour_counts.sum()
        d_sum = day_counts.sum()
        self.hour_probs = (
            hour_counts / h_sum if h_sum > 0
            else np.full(self.N_HOURS, 1.0 / self.N_HOURS)
        )
        self.day_probs = (
            day_counts / d_sum if d_sum > 0
            else np.full(self.N_DAYS, 1.0 / self.N_DAYS)
        )

    # ------------------------------------------------------------------
    # Context weight
    # ------------------------------------------------------------------

    def context_weight(self, now_ts: float) -> float:
        """Scalar in [0.9, 1.1] derived from the user's rhythm at *now_ts*."""
        dt = datetime.datetime.fromtimestamp(now_ts)
        h_prob = float(self.hour_probs[dt.hour])
        d_prob = float(self.day_probs[dt.weekday()])
        combined = self._W_HOUR * h_prob + self._W_DAY * d_prob
        return 0.9 + 0.2 * combined

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-serialisable dict."""
        return {
            "hour_probs": self.hour_probs.tolist(),
            "day_probs": self.day_probs.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ListenRhythm:
        """Reconstruct from a :pymeth:`to_dict` output."""
        return cls(
            hour_probs=np.array(d["hour_probs"], dtype=np.float64),
            day_probs=np.array(d["day_probs"], dtype=np.float64),
        )
