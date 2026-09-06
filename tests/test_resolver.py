"""Tests for the MBID resolver and ListenBrainz importer — pytest-free."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.importer import import_listens
from src.profile import LocalProfile
from src.resolver import MbidResolver, RECORD_SIZE


def _header(name: str) -> None:
    print(f"--- {name} ---")


def _write_index(path: Path, entries: list[tuple[str, int]]) -> None:
    """entries: (mbid_str, item_id), written sorted by UUID bytes."""
    recs = sorted(((uuid.UUID(m).bytes, i) for m, i in entries), key=lambda x: x[0])
    buf = bytearray(len(recs) * RECORD_SIZE)
    for i, (key, iid) in enumerate(recs):
        off = i * RECORD_SIZE
        buf[off : off + 16] = key
        struct.pack_into("<I", buf, off + 16, iid)
    path.write_bytes(bytes(buf))


def test_resolver() -> None:
    _header("test_resolver")
    with tempfile.TemporaryDirectory() as tmp:
        idx = Path(tmp) / "mbid_index.bin"
        mbid_a = "00000000-0000-0000-0000-00000000000a"
        mbid_b = "00000000-0000-0000-0000-00000000000b"
        _write_index(idx, [(mbid_a, 11), (mbid_b, 42)])

        r = MbidResolver(idx)
        assert len(r) == 2
        assert r.resolve(mbid_a) == 11
        assert r.resolve(mbid_b) == 42
        assert r.resolve("00000000-0000-0000-0000-0000000000ff") is None  # absent
        assert r.resolve("not-a-uuid") is None  # invalid
        found, missing = r.resolve_many([mbid_a, mbid_b, "not-a-uuid"])
        assert found == {mbid_a: 11, mbid_b: 42} and missing == 1
    print("PASS")


def test_importer() -> None:
    _header("test_importer")
    with tempfile.TemporaryDirectory() as tmp:
        idx = Path(tmp) / "mbid_index.bin"
        mbid_a = "00000000-0000-0000-0000-00000000000a"
        mbid_b = "00000000-0000-0000-0000-00000000000b"
        _write_index(idx, [(mbid_a, 11), (mbid_b, 42)])
        resolver = MbidResolver(idx)

        export = {
            "listens": [
                {"listened_at": 1000, "track_metadata": {"additional_info": {"recording_mbid": mbid_a}}},
                {"listened_at": 2000, "track_metadata": {"additional_info": {"recording_mbid": mbid_a}}},
                {"listened_at": 3000, "track_metadata": {"mbid": mbid_b}},
                {"listened_at": 4000, "track_metadata": {"additional_info": {"recording_mbid": "ffffffff-ffff-ffff-ffff-ffffffffffff"}}},
                {"listened_at": 5000, "track_metadata": {}},  # no mbid at all
            ]
        }
        f = Path(tmp) / "export.json"
        f.write_text(json.dumps(export))

        profile, stats = import_listens(f, resolver)
        assert stats["n_listens"] == 5
        assert stats["n_resolved"] == 3
        assert stats["n_unresolved"] == 2
        assert stats["n_unique_items"] == 2
        assert profile.item_counts == {11: 2, 42: 1}
        assert profile.last_seen_ts[11] == 2000.0
        assert abs(stats["coverage"] - 0.6) < 1e-9
    print("PASS")


if __name__ == "__main__":
    test_resolver()
    test_importer()
    print("\nResults: 2 PASS, 0 FAIL out of 2")
