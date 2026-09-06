# Serving-Tier Comparison: Raw f32 vs Sonata PQ vs Etude PQ

Same catalog (2,803,656 items × 64-dim item2vec embeddings), same hardware
(Intel i3-12100F, 7.8 GB RAM, pure NumPy runtime, single thread), same
hybrid ranker. PQ tiers scored via ADC (per-subspace lookup tables, no
reconstruction). Sources: `benchmark_raw.json`, `benchmark_pq_sonata.json`,
`benchmark_pq_etude.json`, `sweep_results.json`, Kaggle lb-quantize
`summary.json`.

| Metric | Raw f32 | Sonata PQ (16×) | Etude PQ (32×) |
|---|---|---|---|
| PQ config | — | 16 subspaces × 256 centroids (dim 4) | 8 subspaces × 256 centroids (dim 8) |
| Model size on disk | 684.5 MB | 42.8 MB | 20.9 MB |
| Compression vs f32 | 1× | 15.98× | 31.91× |
| Cold start | 3.24 s | 0.41 s | 0.24 s |
| Recommend p50 | 1,230.1 ms | 722.9 ms | 536.2 ms |
| Recommend p95 | 1,943.0 ms | 802.2 ms | 583.4 ms |
| Peak RSS | 1,829.5 MB | 534.9 MB | 512.2 MB |
| Reconstruction MSE | 0 (exact) | 0.001029 | 0.002904 |
| ANN recall@100 (vs exact top-100) | 1.000 | 0.358 | 0.168 |
| Top-20 Jaccard vs raw tier | 1.000 | 0.333–0.538 | 0.143–0.212 |
| Resolver index (shared) | 53.5 MB | 53.5 MB | 53.5 MB |

## Reading the table

- **Raw f32** is the quality ceiling: exact cosine over the full catalog.
  It costs 1.83 GB peak RSS — infeasible on phones, marginal on this
  desktop's 7.8 GB.
- **Sonata** cuts storage 16× and RSS 3.4× while keeping roughly a third
  to half of the raw tier's top-20 recommendations (Jaccard). ANN
  recall@100 of 0.358 means it finds ~36% of the exact top-100 neighbors.
- **Etude** halves storage again (32×) with only ~22 MB of codes, but
  overlap with the raw tier collapses to 0.14–0.21 and ANN recall@100 to
  0.168. Treat it as an emergency-storage tier, not a quality tier.

## Latency note

PQ tiers are *faster* than raw despite doing more work per item, because
ADC gathers (uint8 code lookups) touch far less memory than streaming a
717 MB float matrix past the CPU. The p95 gap (1.9 s → 0.6–0.8 s) is the
memory-bandwidth bound, not compute.

## Recommendation

Ship **Sonata** as the default on-device tier: 43 MB model, 0.5 GB RSS,
sub-second p95. Keep raw for desktop/server serving where exactness
matters. Etude only where 21 MB is a hard constraint and discovery
quality is acceptable to degrade.
