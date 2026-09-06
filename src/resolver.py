"""MBID -> v1_item_id resolver — binary search over a flat sorted index.

Index file format (models/mbid_index.bin, built by tools/build_resolver_index.py):
    fixed 20-byte records, sorted by the 16-byte MBID key:
    [16-byte raw UUID][4-byte little-endian uint32 item_id]

No pyarrow, no network — stdlib only, ~56 MB in memory for the full vocab.
"""

from __future__ import annotations

import struct
import uuid
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

RECORD_SIZE = 20
KEY_SIZE = 16


class MbidResolver:
    """Resolve recording MBIDs to V1 item ids via binary search."""

    def __init__(self, index_path: str | Path) -> None:
        self._blob = Path(index_path).read_bytes()
        if len(self._blob) % RECORD_SIZE:
            raise ValueError(f"index size {len(self._blob)} not a multiple of {RECORD_SIZE}")
        self.n_items = len(self._blob) // RECORD_SIZE

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def _search(self, key: bytes) -> Optional[int]:
        lo, hi = 0, self.n_items
        while lo < hi:
            mid = (lo + hi) // 2
            k = self._blob[mid * RECORD_SIZE : mid * RECORD_SIZE + KEY_SIZE]
            if k < key:
                lo = mid + 1
            else:
                hi = mid
        if lo < self.n_items:
            off = lo * RECORD_SIZE
            if self._blob[off : off + KEY_SIZE] == key:
                return struct.unpack_from("<I", self._blob, off + KEY_SIZE)[0]
        return None

    def resolve(self, mbid: str) -> Optional[int]:
        """Return the v1_item_id for a MBID string, or None if unknown."""
        try:
            key = uuid.UUID(mbid).bytes
        except (ValueError, AttributeError):
            return None
        return self._search(key)

    def resolve_many(
        self, mbids: Iterable[str]
    ) -> Tuple[dict[str, int], int]:
        """Resolve a batch. Returns ({mbid: item_id}, n_unresolved)."""
        found: dict[str, int] = {}
        missing = 0
        for m in mbids:
            iid = self.resolve(m)
            if iid is None:
                missing += 1
            else:
                found[m] = iid
        return found, missing

    def __len__(self) -> int:
        return self.n_items
