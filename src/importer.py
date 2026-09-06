"""Import a ListenBrainz listen export into a LocalProfile.

Usage:
    python -m src.importer export.json profile.json

Accepts the ListenBrainz export/API shape: {"listens": [{"listened_at": ...,
"track_metadata": {...}}, ...]}. Recording MBID is read from
track_metadata.additional_info.recording_mbid, falling back to
track_metadata.mbid. Unresolvable MBIDs (not in the vocab) are dropped
and counted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

from .profile import LocalProfile
from .resolver import MbidResolver


def _extract_mbid(listen: dict) -> str | None:
    tm = listen.get("track_metadata") or {}
    addl = tm.get("additional_info") or {}
    return addl.get("recording_mbid") or tm.get("mbid") or None


def import_listens(
    export_path: str | Path, resolver: MbidResolver
) -> Tuple[LocalProfile, dict]:
    """Convert a ListenBrainz export into a (LocalProfile, stats)."""
    raw = json.loads(Path(export_path).read_text(encoding="utf-8"))
    listens = raw.get("listens", raw) if isinstance(raw, dict) else raw

    profile = LocalProfile()
    n_listens = n_resolved = n_unresolved = 0
    for listen in listens:
        n_listens += 1
        mbid = _extract_mbid(listen)
        ts = float(listen.get("listened_at", 0))
        if mbid is None:
            n_unresolved += 1
            continue
        iid = resolver.resolve(mbid)
        if iid is None:
            n_unresolved += 1
        else:
            profile.observe(iid, ts)
            n_resolved += 1

    stats = {
        "n_listens": n_listens,
        "n_resolved": n_resolved,
        "n_unresolved": n_unresolved,
        "n_unique_items": len(profile.item_counts),
        "coverage": (n_resolved / n_listens) if n_listens else 0.0,
    }
    return profile, stats


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: python -m src.importer <export.json> <profile.json>")
        sys.exit(1)
    export, out = sys.argv[1], sys.argv[2]

    models_dir = Path(__file__).resolve().parent.parent / "models"
    resolver = MbidResolver(models_dir / "mbid_index.bin")
    profile, stats = import_listens(export, resolver)

    Path(out).write_text(json.dumps(profile.to_dict()), encoding="utf-8")
    print(f"imported {stats['n_resolved']:,}/{stats['n_listens']:,} listens "
          f"({stats['coverage']:.1%} coverage) -> {stats['n_unique_items']:,} unique items")
    print(f"profile written: {out}")


if __name__ == "__main__":
    main()
