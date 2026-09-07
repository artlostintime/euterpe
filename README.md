# music-recommender

Local-first music recommendation engine: a hybrid ranker that fuses a
per-user repeat engine (decayed, signal-weighted play counts) with a
catalog discovery engine (item2vec embedding similarity, PQ-compressed,
ADC-scored). Trained and evaluated on MLHD+ (1.68B listening events,
36,970 users, 2.8M-item catalog).

## Architecture

- **Repeat engine** — decayed frequency over the user's own history.
  Learns online from every event; no training loop.
- **Discovery engine** — cosine similarity between the user's
  decay-weighted taste center and catalog embeddings. Served from
  product-quantized codes via asymmetric distance computation (no
  reconstruction).
- **Context weighting** — hour-of-day / day-of-week listening rhythm
  histograms, clamped to [0.9, 1.1].
- **Signal weighting** — play / completion / replay / like / save /
  playlist_add / short_play / skip, each with its own weight.
- **Exploration preference** — user-tunable repeat/discovery blend.

## Install

Python 3.10+, numpy only:

```bash
pip install numpy
```

Optional: pyarrow (only for `tools/build_resolver_index.py`).

## Model weights

The catalog embeddings and serving artifacts are hosted on HuggingFace:
**https://huggingface.co/<user>/music-recommender-models** (placeholder —
set your HF repo URL after creating it).

| File | Size | Purpose |
|---|---|---|
| `item2vec_final.npy` | 684.5 MB | raw f32 embeddings (2,803,656 x 64) |
| `sonata_pq.npz` | 42.8 MB | PQ tier, 16x compression (default) |
| `etude_pq.npz` | 20.9 MB | PQ tier, 32x compression (low storage) |
| `mbid_index.bin` | 53.5 MB | MBID -> item_id binary-search index |
| `v1_vocab.parquet` | 58.4 MB | item vocabulary (MBID, count, id) |
| `top_items.json` | 0.2 MB | cold-start popularity + name bridge |

Download into `models/` (gitignored). The runtime discovers them
automatically from `models/` or the search paths in `src/ranker.py`.

## Usage

```bash
# self-contained demo (builds a profile from top_items.json)
python -m src --demo

# import a ListenBrainz export and serve recommendations
python -m src.importer export.json profile.json
python -m src --profile profile.json

# opt-in minimized contribution payload (counts only, no raw timestamps)
python -m src --contribute
```

## Tests

26 tests, no framework required:

```bash
python tests/test_runtime.py      # 9
python tests/test_resolver.py     # 2
python tests/test_context.py      # 6
python tests/test_contribution.py # 4
python tests/test_signals.py      # 5
```

## Benchmarks (i3-12100F, 7.8 GB RAM, pure NumPy)

| Tier | Model size | Cold start | p50 / p95 | Peak RSS |
|---|---|---|---|---|
| Raw f32 | 684.5 MB | 3.24 s | 1230 / 1943 ms | 1,830 MB |
| Sonata PQ (16x) | 42.8 MB | 0.41 s | 723 / 802 ms | 535 MB |
| Etude PQ (32x) | 20.9 MB | 0.24 s | 536 / 583 ms | 512 MB |

Full comparison: `reports/tier_comparison.md`. Config sweep:
`reports/sweep_results.json`.

## Repository layout

```
src/        profile, ranker, resolver, importer, context, contribution
tools/      benchmark, sweep, build_resolver_index
tests/      26 tests (pytest-free)
docs/       app-integration.md (embedding in a streaming app)
reports/    benchmarks, tier comparison, sweep results
models/     model files (gitignored — download from HuggingFace)
```

## Provenance

- Embeddings trained on MLHD+ (MetaBrainz Foundation, CC0).
- Evaluation protocol and results: see the companion paper
  "Why Did the Algorithm Think I'd Like This?" (Shuvi, 2026).
- Dataset audit: "Sanitizing Music Listening Histories at Scale"
  (Shuvi, 2026), Zenodo DOI 10.5281/zenodo.22338293.

## License

Apache 2.0 (see LICENSE). Model weights: Apache 2.0, same terms.
