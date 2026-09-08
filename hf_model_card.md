---
license: apache-2.0
library_name: modelcard
tags:
- music
- recommendation
- item2vec
- product-quantization
- local-first
- on-device
- mlhd
datasets:
- MetaBrainz/Music-Listening-Histories-Dataset-Plus
---

# euterpe-model — serving artifacts for the Euterpe music recommender

Model weights and serving artifacts for **Euterpe**, a local-first music
recommendation runtime. Trained on MLHD+ (Music Listening Histories
Dataset Plus, 1.68 billion sanitized listening events, 36,970 users,
2.8M recordings). See the runtime repo:
[artlostintime/euterpe](https://github.com/artlostintime/euterpe).

## What this is

Two engines, one ranker:

- **Repeat engine** — decayed-frequency scoring over the user's own
  history. Exact, instant, runs from the profile alone (no model file).
- **Discovery engine** — item2vec embeddings over the full catalog,
  scored by cosine similarity via asymmetric distance computation (ADC)
  on product-quantized codes. This repo ships those codes.

The hybrid ranker fuses both, with an explicit repeat/discovery weight
and optional hour-of-day / day-of-week context weighting.

## Files

| File | Size | Purpose |
|---|---|---|
| `item2vec_final.npy` | 684.5 MB | Raw f32 embeddings (2,803,656 × 64), exact |
| `sonata_pq.npz` | 42.8 MB | PQ codes, 16 subspaces — 16× compression, ANN recall@100 0.358 |
| `etude_pq.npz` | 20.9 MB | PQ codes, 8 subspaces — 32× compression, ANN recall@100 0.168 |
| `mbid_index.bin` | 53.5 MB | MBID → item_id resolver (binary-search index, 2.8M rows) |
| `v1_vocab.parquet` | 58 MB | recording_mbid → item_id vocabulary + counts |
| `top_items.json` | 0.2 MB | Cold-start popularity ranking + name bridge (top 2,000) |
| `metrics.json` | — | Unified evaluation metrics (see below) |

## Serving benchmarks

Intel i3-12100F, 7.8 GB RAM, pure NumPy runtime, no GPU:

| Tier | Model size | Cold start | Recommend p50 | Peak RSS |
|---|---|---|---|---|
| Raw f32 | 684.5 MB | 3.24 s | 1,230 ms | 1.83 GB |
| Sonata PQ (16×) | 42.8 MB | 0.41 s | 723 ms | 535 MB |
| Etude PQ (32×) | 20.9 MB | 0.24 s | 536 ms | 512 MB |

Sonata is the recommended default tier; etude is the low-storage tier.

## Evaluation

Measured on a 5,000-user held-out cohort under a per-user temporal
protocol (validation-based hyperparameter selection, test with
train+validation history, train-only candidate universe of 2,787,934
items). Headline findings (recall@100, %):

- Repeats: decayed-frequency scoring dominates among evaluated
  scorers — decay_30d 28.9% vs user-frequency 12.9%; the
  validation-selected fusion matches the best single scorer (29.0%).
- Discovery: embedding similarity leads — item2vec with a 10-item
  profile window reaches 1.73% vs 0.47% for the popularity baseline;
  frequency-based scorers saturate at zero (empirical saturation:
  the average user's top-500 recommendations are ≥99.99% history
  items).
- BPR-MF (evaluated configuration) is statistically indistinguishable
  from popularity overall (+0.09pp, 95% CI [-0.02, +0.20]) and lower
  on the discovery split.
- Robustness: conclusions stable across cohort seeds 42/43/44 (max sd
  0.36pp) and under a global-chronological split sensitivity check.
- Full metrics: 22 scorers × recall/precision/NDCG@20/@100, MRR@10,
  hitrate@20, novelty, diversity, per-user percentiles, paired
  bootstrap CIs, fusion weight sweep, catalog coverage.

## Usage

```python
# with the euterpe runtime (github.com/artlostintime/euterpe)
from src.ranker import HybridRanker
from src.profile import LocalProfile

ranker = HybridRanker(pq_path="sonata_pq.npz")   # ADC scoring, no reconstruction
profile = LocalProfile()
profile.observe(item_id, timestamp, signal="play")  # or "like", "save", ...
recs = ranker.recommend(profile, k=20)
```

## Training

Embeddings trained with item2vec (SGNS) on sanitized MLHD+ co-listening
events; PQ codebooks trained with k-means (k=256, 15 Lloyd iterations,
200k-item sample, seed 42). Full pipeline reproducible via the versioned
Kaggle kernels in the runtime repo (deterministic, checksum-verified).

## Limitations

- Discovery recall is modest (best 1.73% recall@100 on a 2.8M-item
  catalog) — should not be interpreted as an estimate of
  product-level recommendation performance.
- Embeddings cover the MLHD+ catalog as of the training snapshot; new
  releases need the popularity fallback until the next retrain.
- The cohort is MLHD+ power users; repeat rates on general populations
  will differ.
- No online/serving evaluation; feedback-loop effects are out of scope.

## Citation

```bibtex
@misc{shuvi2026euterpe,
  author = {Shuvi},
  title = {Euterpe: local-first music recommendation runtime and serving artifacts},
  year = {2026},
  url = {https://huggingface.co/artlostintime/euterpe-model}
}
```

See also the companion papers: an MLHD+ dataset audit (JOHD submission,
Zenodo DOI 10.5281/zenodo.22338293) and a behavioral-signals
recommendation study (JOHD submission, Zenodo DOI
10.5281/zenodo.22661887).

## License

Apache 2.0. Embeddings derived from MLHD+ (CC0 1.0, MetaBrainz
Foundation).
