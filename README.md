# music-recommender runtime skeleton

Minimal on-device recommendation ranker: local profile + global embeddings + recency decay.

## Run demo

```bash
cd D:\music-recommender
python -m src --demo
```

Requires numpy. Optional: place `.npy` (2-D float) or `.npz` (keys: `codes`, `centroids`) embedding files under `models/` or the temp `lb/` path for the full demo.

## Run tests

```bash
python tests\test_runtime.py
```

All tests use synthetic data — no external files needed.

## Scope

Research skeleton for local-first hybrid ranking (Phase 11-16 ground-work).
Not mobile-ready — raw embeddings are 717 MB; PQ compression reduces this but inference is still CPU-bound.
