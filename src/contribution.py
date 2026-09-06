"""Phase 16 — opt-in global contribution builder.

Privacy-filtered export artifact for voluntary contribution to global
model training.  No network, no upload — just the export payload.
"""

from __future__ import annotations

import datetime
from typing import Dict

from .profile import LocalProfile


class ContributionBuilder:
    """Build a minimised, privacy-filtered contribution payload from a profile.

    Consent must be explicitly set to True — every ``build()`` call
    raises ValueError otherwise.  This is the opt-in gate.
    """

    def __init__(self, profile: LocalProfile, consent: bool = False) -> None:
        self._profile = profile
        self._consent = consent

    def _check(self) -> None:
        if not self._consent:
            raise ValueError(
                "Consent required: set consent=True to build a contribution payload"
            )

    def build(
        self,
        max_items: int = 200,
        min_count: int = 5,
        coarse_ts: bool = True,
    ) -> dict:
        """Produce a minimised contribution payload.

        Filtering:
        - Only items with ``count >= min_count`` (removes rare one-offs).
        - Capped to ``max_items`` by highest count (bounded payload).
        - If ``coarse_ts``, exact timestamps are replaced with month
          buckets (``"YYYY-MM"``) to preserve seasonal signal without
          precise timing.
        """
        self._check()

        eligible = sorted(
            ((iid, cnt) for iid, cnt in self._profile.item_counts.items()
             if cnt >= min_count),
            key=lambda x: -x[1],
        )[:max_items]

        items: list[dict] = []
        for iid, cnt in eligible:
            entry: Dict[str, object] = {"item_id": iid, "count": cnt}
            if coarse_ts:
                ts = self._profile.last_seen_ts.get(iid, 0.0)
                dt = datetime.datetime.fromtimestamp(ts)
                entry["last_seen_month"] = f"{dt.year:04d}-{dt.month:02d}"
            items.append(entry)

        return {
            "schema": "contribution.v1",
            "n_items": len(items),
            "items": items,
        }
