# AttnRes Reproduction: Results Log

Findings from blessed runs. All results are relative baseline-vs-AttnRes comparisons at matched config.

## 30M (off-curve, 6 layers, 3 blocks) — 3 seeds, 150M tokens each

### Perplexity (claim 1: AttnRes improves over baseline) — REPRODUCED
Final val perplexity (3 seeds):
- Baseline: 122.19, 120.98, 118.54 (mean ~120.6)
- Block AttnRes: 105.05, 104.23, 100.93 (mean ~103.4)
- Full AttnRes: 103.86, 101.65, 95.47 (mean ~100.3)

Both AttnRes variants beat baseline by ~15-17%. Clean non-overlapping separation between baseline cluster and AttnRes cluster across all seeds. Full slightly ahead of Block (overlapping, expected — Block recovers most of Full's benefit per the paper).

### Stabilization (claim 3: AttnRes bounds hidden-state magnitude) — REPRODUCED
Fig 5b (per-layer h_l magnitude across depth, final checkpoint, 3 seeds):
- Baseline grows with depth, peaks ~17 (PreNorm dilution).
- Full AttnRes bounded low throughout (~1.7-5, convex combination).
- Block AttnRes intermediate (~4-9) with visible reset/sawtooth at block boundary.
All three paper behaviors present. Sawtooth coarse at 30M (only 3 blocks); expected clearer at ladder sizes with more blocks. Baseline mid-peak-then-dip is a short-model (6-layer) artifact.

Last-layer magnitude (final, 3 seeds): baseline ~11.6, Block ~4.6, Full ~1.7. Clean seed separation.

### Throughput note
Block faster than Full (~49.5k vs ~40.3k tok/s at 30M) — consistent with Block attending over fewer sources (its efficiency advantage).

## 90M (off-curve, 12 layers, 6 blocks) — 2 seeds, in progress
[pending]

## Key methodological notes
- All models dense decoder-only; original AttnRes is entirely MoE. Dense reproduction = generalization contribution.
- Ladder holds d/L ~73 constant (matches Moonshot Table 2). Small off-curve models (30M, 90M) shaped for meaningful blocks, not on scaling curve.
- Original used ~8 blocks approximately (per-model counts unspecified, layer counts 12-17 don't all divide by 8); we fix block counts precisely per size for controlled comparison.