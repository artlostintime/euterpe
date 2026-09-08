# Research kernels

Deterministic Kaggle kernels for the full train/eval pipeline. Each
folder contains the current source as pushed to Kaggle
(`kaggle kernels pull shuvitobe/<name>`); each is self-contained and
pushable as-is.

## Chain

```
lb-sanitize -> lb-eda -> lb-trainprep -> lb-item2vec / lb-ranker / lb-bpr
                       -> lb-eval (22-scorer unified protocol) -> lb-quantize
```

Audit-side kernels (Paper 1): `lb-phase1`, `lb-census`, `lb-crosscheck`,
`lb-eyeball`, `lb-recency`, `lb-sensitivity`.

## Kernels

| Kernel | Purpose |
|---|---|
| `lb-sanitize` | MLHD+ sanitization (spam/loop/bot filtering) |
| `lb-eda` | exploratory analysis of sanitized events |
| `lb-census` | dataset census (users, items, events, sparsity) |
| `lb-crosscheck` | cross-check audit of sanitization decisions |
| `lb-eyeball` | manual-sample inspection harness |
| `lb-trainprep` | temporal splits, candidate universe, cohorts |
| `lb-item2vec` | SGNS item2vec training (64-dim embeddings) |
| `lb-ranker` | decayed-repeat ranker + fusion weights |
| `lb-bpr` | BPR-MF baseline training |
| `lb-recency` | recency-decay scorer comparison |
| `lb-sensitivity` | split/parameter sensitivity analysis |
| `lb-eval` | unified evaluation, 22 scorers (v5.0.0) |
| `lb-quantize` | product quantization (sonata 16x / etude 32x) |
| `lb-phase1` | initial feasibility pass (superseded) |

## Frozen versions vs. this repo

This folder is the **living** pipeline. The exact kernel versions that
produced the published results are frozen in the Zenodo archives:

- Paper 1 (MLHD+ audit): Zenodo DOI 10.5281/zenodo.22338293
- Paper 2 (behavioral signals): evaluator lb-eval v5.0.0 + evidence v6,
  Zenodo DOI 10.5281/zenodo.22661887

Paper 2 benchmark provenance: evaluator lb-eval v5.0.0, evidence
release v6, primary cohort seed 42, additional seeds 43/44.

Every reported number in the papers comes from these kernels under
frozen seeds.
