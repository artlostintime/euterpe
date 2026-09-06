"""Latency / RAM / size benchmark for the hybrid ranker (BUILD spec Phase 10).

Usage:
    python tools/benchmark.py [models_dir] [--tier raw|pq]

Measures: cold-start (index + embeddings load), warm recommend() p50/p95,
peak working-set RSS (Windows via ctypes), model bytes on disk.
Writes reports/benchmark_<tier>.json next to the repo root.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.profile import LocalProfile
from src.ranker import HybridRanker


class PeakRSS:
    """Windows peak working-set tracker via GetProcessMemoryInfo (stdlib ctypes)."""

    class _PMC(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    def __init__(self) -> None:
        self._pmc = self._PMC()
        self._pmc.cb = ctypes.sizeof(self._PMC)
        self._h = ctypes.windll.kernel32.GetCurrentProcess()
        self._f = ctypes.windll.psapi.GetProcessMemoryInfo
        self._f.argtypes = [wt.HANDLE, ctypes.POINTER(self._PMC), wt.DWORD]

    def peak_mb(self) -> float:
        self._f(self._h, ctypes.byref(self._pmc), self._pmc.cb)
        return self._pmc.PeakWorkingSetSize / (1024 * 1024)


def build_profile(models_dir: Path) -> LocalProfile:
    """Real-data profile from top_items.json (same rule as the demo)."""
    import random
    rng = random.Random(42)
    now = time.time()
    p = LocalProfile()
    top = json.loads((models_dir / "top_items.json").read_text())
    for entry in top[::6][:30]:
        for _ in range(rng.randint(1, 10)):
            p.observe(entry["v1_item_id"], now - rng.uniform(0, 90 * 86400))
    return p


def main() -> None:
    models_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).resolve().parent.parent / "models"
    tier = "pq" if "--tier" in sys.argv and sys.argv[sys.argv.index("--tier") + 1] == "pq" else "raw"

    rss = PeakRSS()
    base_mb = rss.peak_mb()

    # -- cold start --
    t0 = time.time()
    emb_path, pq_path = HybridRanker.discover_embeddings()
    ranker = HybridRanker(embeddings_path=emb_path, pq_path=pq_path)
    assert ranker.has_embeddings(), "no embeddings found"
    cold_s = time.time() - t0

    profile = build_profile(models_dir)

    # -- warm recommend --
    lat = []
    for i in range(20):
        t = time.time()
        ranker.recommend(profile, k=20)
        lat.append((time.time() - t) * 1000.0)
    lat.sort()
    p50 = statistics.median(lat)
    p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]

    # -- sizes --
    src = pq_path if pq_path else emb_path
    model_mb = src.stat().st_size / (1024 * 1024)
    idx = models_dir / "mbid_index.bin"
    idx_mb = idx.stat().st_size / (1024 * 1024) if idx.exists() else 0.0

    result = {
        "tier": tier, "source": str(src),
        "cold_start_s": round(cold_s, 2),
        "recommend_p50_ms": round(p50, 1),
        "recommend_p95_ms": round(p95, 1),
        "peak_rss_mb": round(rss.peak_mb(), 1),
        "baseline_rss_mb": round(base_mb, 1),
        "model_mb": round(model_mb, 1),
        "resolver_index_mb": round(idx_mb, 1),
        "n_items": int(ranker._embeddings.shape[0]),
    }
    out = Path(__file__).resolve().parent.parent / "reports"
    out.mkdir(exist_ok=True)
    (out / f"benchmark_{tier}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
