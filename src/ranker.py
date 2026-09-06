"""Hybrid ranker — fuses repeat and discovery signals."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .profile import LocalProfile

# ------------------------------------------------------------------
# PQ decode helpers
# ------------------------------------------------------------------

# ponytail: global lock for PQ decode; fine at single-thread on-device
# Upgrade: per-subspace parallel decode if throughput matters


def _pq_decode(codes: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Reconstruct float vectors from PQ codes.

    Args:
        codes: (n_items, n_subspaces) uint8
        centroids: (n_subspaces, k, sub_dim)

    Returns:
        (n_items, n_subspaces * sub_dim) float32
    """
    n_sub, k, sub_dim = centroids.shape
    out = np.empty((codes.shape[0], n_sub * sub_dim), dtype=np.float32)
    for s in range(n_sub):
        out[:, s * sub_dim : (s + 1) * sub_dim] = centroids[s, codes[:, s]]
    return out


# ------------------------------------------------------------------
# Score helpers
# ------------------------------------------------------------------

def _min_max(vals: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1].  Returns zeros if constant."""
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-12:
        return np.zeros_like(vals)
    return (vals - lo) / (hi - lo)


# ------------------------------------------------------------------
# Ranker
# ------------------------------------------------------------------

class HybridRanker:
    """Fuse repeat (decay) + discovery (embedding cosine) signals.

    Supports raw .npy embedding matrices and .npz PQ-compressed codes.
    Embeddings are loaded via mmap when possible.
    """

    # Embedding search paths tried by --demo
    SEARCH_DIRS: list[str] = [
        r"D:\music-recommender\models",
        r"C:\Users\assas\AppData\Local\Temp\opencode\lb",
    ]

    def __init__(
        self,
        embeddings_path: str | Path | None = None,
        pq_path: str | Path | None = None,
        w_repeat: float = 0.7,
        w_discovery: float = 0.3,
    ) -> None:
        self.w_repeat = w_repeat
        self.w_discovery = w_discovery

        self._embeddings: Optional[np.ndarray] = None  # (n, d) float32
        self._item_ids: Optional[np.ndarray] = None     # (n,) int
        self._embeddings_path: Optional[str | Path] = embeddings_path
        self._pq_path: Optional[str | Path] = pq_path
        self._loaded = False

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._loaded:
            return
        if self._pq_path and Path(self._pq_path).exists():
            data = np.load(self._pq_path)
            codes = data["codes"]         # (n, n_sub) uint8
            centroids = data["centroids"] # (n_sub, k, sub_dim)
            self._embeddings = _pq_decode(codes, centroids)
            self._item_ids = (
                data["item_ids"] if "item_ids" in data
                else np.arange(self._embeddings.shape[0])
            )
        elif self._embeddings_path and Path(self._embeddings_path).exists():
            # Ponytail: load fully, not mmap — we need all rows for cosine
            # anyway, and mmap holds Windows file locks.
            self._embeddings = np.load(self._embeddings_path).astype(np.float32)
            self._item_ids = np.arange(self._embeddings.shape[0])
        else:
            self._embeddings = None
            self._item_ids = None
        self._loaded = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_embeddings(self) -> bool:
        self._load()
        return self._embeddings is not None

    def recommend(
        self,
        profile: LocalProfile,
        k: int = 20,
        exclude_played: bool = True,
        half_life_days: float = 90.0,
    ) -> List[int]:
        """Return *k* item ids ranked by hybrid score.

        repeat signal  = decay profile scores
        discovery signal = cosine(sim(profile_mean_emb, item_emb))
        """
        self._load()

        # --- repeat scores ---
        repeat_raw = profile.decay_scores(half_life_days=half_life_days)
        played_ids = set(profile.item_counts.keys())

        if self._embeddings is not None and len(repeat_raw) > 0:
            id_to_idx = {int(iid): idx for idx, iid in enumerate(self._item_ids)}
            # Build repeat score vector aligned with embeddings
            repeat_arr = np.zeros(self._embeddings.shape[0], dtype=np.float32)
            for iid, sc in repeat_raw.items():
                if iid in id_to_idx:
                    repeat_arr[id_to_idx[iid]] = sc
        else:
            # No embeddings: just return repeat-ranked items
            ranked = sorted(repeat_raw.items(), key=lambda x: -x[1])
            return [iid for iid, _ in ranked[:k]]

        # --- discovery scores (cosine sim) ---
        played_emb_ids = [iid for iid in played_ids if iid in id_to_idx]
        if len(played_emb_ids) > 0:
            played_vecs = self._embeddings[[id_to_idx[iid] for iid in played_emb_ids]]
            profile_mean = played_vecs.mean(axis=0)
            # cosine sim = dot / (||a|| * ||b||), normalised by mean of item norms
            norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-12
            centered = self._embeddings / norms
            p_norm = np.linalg.norm(profile_mean) + 1e-12
            discovery_arr = (centered @ profile_mean) / p_norm  # (n,)
        else:
            discovery_arr = np.zeros(self._embeddings.shape[0], dtype=np.float32)

        # --- exclude played ---
        mask = np.ones(self._embeddings.shape[0], dtype=bool)
        if exclude_played:
            for iid in played_ids:
                if iid in id_to_idx:
                    mask[id_to_idx[iid]] = False

        # --- normalise & fuse ---
        repeat_norm = _min_max(repeat_arr)
        discovery_norm = _min_max(discovery_arr)

        hybrid = self.w_repeat * repeat_norm + self.w_discovery * discovery_norm
        hybrid[~mask] = -1.0  # push played items to bottom

        top_k_idx = np.argpartition(-hybrid, min(k, hybrid.size - 1))[:k]
        # stable sort for deterministic output
        top_k_idx = top_k_idx[np.argsort(-hybrid[top_k_idx])]

        return [int(self._item_ids[i]) for i in top_k_idx]

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @classmethod
    def discover_embeddings(cls) -> Tuple[Optional[Path], Optional[Path]]:
        """Search common paths for .npy or PQ .npz files.

        Returns (embeddings_path, pq_path) — at most one will be non-None.
        """
        for d in cls.SEARCH_DIRS:
            base = Path(d)
            if not base.exists():
                continue
            # PQ files first (smaller)
            for f in base.rglob("*.npz"):
                try:
                    data = np.load(f)
                    if "codes" in data and "centroids" in data:
                        return (None, f)
                except Exception:
                    pass
            # .npy 2-D float files
            for f in base.rglob("*.npy"):
                try:
                    arr = np.load(f, mmap_mode="r")
                    if arr.ndim == 2 and np.issubdtype(arr.dtype, np.floating):
                        return (f, None)
                except Exception:
                    pass
        return (None, None)
