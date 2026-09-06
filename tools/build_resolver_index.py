"""One-time prep: build models/mbid_index.bin from v1_vocab.parquet.

Usage:
    python tools/build_resolver_index.py [models_dir]

Reads v1_vocab.parquet (pyarrow — prep tool only, runtime stays pyarrow-free)
and writes mbid_index.bin: 20-byte records [16B UUID][4B uint32 item_id],
sorted by UUID bytes. ~56 MB for the full 2.8M-item vocab.
"""

from __future__ import annotations

import struct
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    models_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "models"
    vocab_path = models_dir / "v1_vocab.parquet"
    out_path = models_dir / "mbid_index.bin"

    t0 = time.time()
    t = pq.read_table(str(vocab_path), columns=["recording_mbid", "v1_item_id"])
    mbids = t.column("recording_mbid").to_pylist()
    ids = t.column("v1_item_id").to_pylist()
    print(f"loaded {len(mbids):,} rows in {time.time() - t0:.1f}s")

    records = sorted(zip(mbids, ids), key=lambda x: x[0])
    buf = bytearray(len(records) * 20)
    for i, (mbid, iid) in enumerate(records):
        off = i * 20
        buf[off : off + 16] = mbid
        struct.pack_into("<I", buf, off + 16, iid)

    out_path.write_bytes(bytes(buf))
    print(f"wrote {out_path} ({len(buf):,} bytes) in {time.time() - t0:.1f}s total")


if __name__ == "__main__":
    main()
