# euterpe

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](requirements.txt)
[![Tests](https://img.shields.io/badge/tests-26%2F26-brightgreen)](tests/)
[![Models](https://img.shields.io/badge/models-HuggingFace-yellow)](https://huggingface.co/artlostintime)

Local-first music recommendation: a hybrid engine that fuses a per-user
repeat engine (decayed, signal-weighted play counts) with a catalog
discovery engine (item2vec embeddings, product-quantized, ADC-scored).
Trained and evaluated on MLHD+ — 1.68 billion listening events from
36,970 users over a 2.8M-item catalog.

## Install

```bash
pip install numpy        # only dependency
python -m src --demo     # verify: builds a profile, serves 10 recs
```

Model weights are hosted separately (see [Model weights](#model-weights));
the demo works without them via repeat ranking.

## Quickstart

```python
import sys; sys.path.insert(0, ".")
from src.profile import LocalProfile
from src.ranker import HybridRanker
import time

profile = LocalProfile()
now = time.time()
profile.observe(125, now - 2 * 86400, signal="play")    # Don't Look Back in Anger
profile.observe(125, now - 1 * 86400, signal="replay")
profile.observe(256, now - 5 * 86400, signal="like")    # Somewhere Only We Know

ranker = HybridRanker()                                  # finds PQ weights in models/
recs = ranker.recommend(profile, k=10)                  # -> ranked item ids
```

## How it works

Two engines, fused by an explicit weight:

1. **Repeat engine** — exponentially-decayed, signal-weighted counts over
   the user's own history. Learns online from every event; no training
   loop, fully local. Carries ~75% of measured recommendation quality.
2. **Discovery engine** — cosine similarity between the user's
   decay-weighted taste center and catalog embeddings, served from
   product-quantized codes via asymmetric distance computation (ADC):
   2.8M items in 21-43 MB, scored without materializing the matrix.
3. **Context weighting** — hour/day listening-rhythm histograms,
   clamped to [0.9, 1.1] so context nudges but never overrides behavior.

Signals: `play` (1.0), `completion` (1.2), `replay` (1.0), `like` (2.0),
`save` (1.5), `playlist_add` (1.5), `short_play` (0.3), `skip` (0.0).

## Model weights

Hosted on HuggingFace (Apache 2.0, derived from CC0 MLHD+ data):

| File | Size | Purpose |
|---|---|---|
| `sonata_pq.npz` | 42.8 MB | PQ tier, 16x compression — **default** |
| `etude_pq.npz` | 20.9 MB | PQ tier, 32x compression — low storage |
| `item2vec_final.npy` | 684.5 MB | raw f32 embeddings (2,803,656 x 64) |
| `mbid_index.bin` | 53.5 MB | MBID -> item_id binary-search index |
| `v1_vocab.parquet` | 58.4 MB | item vocabulary (MBID, count, id) |
| `top_items.json` | 0.2 MB | cold-start popularity + name bridge |

Download into `models/` (gitignored); the runtime discovers them
automatically.

## Benchmarks

Intel i3-12100F, 7.8 GB RAM, pure NumPy, no GPU:

| Tier | Model size | Cold start | p50 / p95 | Peak RSS |
|---|---|---|---|---|
| Raw f32 | 684.5 MB | 3.24 s | 1230 / 1943 ms | 1,830 MB |
| Sonata PQ (16x) | 42.8 MB | 0.41 s | 723 / 802 ms | 535 MB |
| Etude PQ (32x) | 20.9 MB | 0.24 s | 536 / 583 ms | 512 MB |

Full comparison: [`reports/tier_comparison.md`](reports/tier_comparison.md).

## Tradeoffs

Honest limits, measured:

- **Discovery is weak in absolute terms** (~1.1% recall@100 on a 2.8M
  catalog with a 14-day test window) — a lower bound on a hard task, not
  a ceiling. The repeat engine is the workhorse; discovery is the only
  path to anything new.
- **PQ costs recall**: sonata retains 35.8%, etude 16.8% of exact
  top-100 neighbors. Quantize the discovery engine, never the repeat
  engine.
- **Cohort scope**: weights and benchmarks reflect MLHD+ power users
  (2005-2013 listening); general populations will differ.
- **No online evaluation**: all numbers are offline. Serving feedback
  loops are documented in `docs/app-integration.md` but not measured.

## Usage

```bash
python -m src --demo                        # self-contained demo
python -m src.importer export.json profile.json   # ListenBrainz export -> profile
python -m src --profile profile.json        # serve recommendations
python -m src --contribute                  # opt-in minimized payload
```

## Tests

26 tests, no framework required:

```bash
python tests/test_runtime.py && python tests/test_resolver.py && \
python tests/test_context.py && python tests/test_contribution.py && \
python tests/test_signals.py
```

## Reproducing the pipeline

The full train/eval pipeline is a chain of deterministic Kaggle kernels
(`kernels/` — each folder is pushable as-is):

```
lb-sanitize -> lb-eda -> lb-trainprep -> lb-item2vec / lb-ranker / lb-bpr
            -> lb-eval (21-scorer unified protocol) -> lb-quantize
```

Every reported number in the papers comes from these kernels under
frozen seeds. See `kernels/README.md` for the chain, runtimes, and
outputs.

## How to cite

If you use euterpe, please cite the companion papers:

```bibtex
@misc{shuvi2026sanitizing,
  title  = {Sanitizing Music Listening Histories at Scale:
            A Reproducible Quality Audit of MLHD+},
  author = {Shuvi},
  year   = {2026},
  note   = {Submitted. Audit archive: Zenodo, DOI 10.5281/zenodo.22338293}
}

@misc{shuvi2026behavioral,
  title  = {Why Did the Algorithm Think I'd Like This?
            Understanding Behavioral Signals in Personalized Music
            Recommendation},
  author = {Shuvi},
  year   = {2026},
  note   = {In submission}
}
```

## Repository layout

```
src/        serving engine (profile, ranker, resolver, importer, context, contribution)
kernels/    reproducible Kaggle pipeline (sanitize -> ... -> quantize)
tools/      benchmark, sweep, resolver-index builder
tests/      26 tests (pytest-free)
docs/       app-integration guide, architecture, reproducibility
reports/    benchmarks, tier comparison, sweep results
models/     model files (gitignored — download from HuggingFace)
```

## License

Apache 2.0 — see [LICENSE](LICENSE). Model weights: Apache 2.0, same
terms. Upstream MLHD+ data: CC0, MetaBrainz Foundation.
