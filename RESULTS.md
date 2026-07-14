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

## 90M (off-curve, 12 layers, 6 blocks) — 2 seeds, ~459M tokens each

### Downstream benchmarks (final checkpoints, lm-eval panel)
Device: Apple Silicon **mps** (torch 2.12.1). Eval compute ≈ **2.7 h** summed across 15 runs. Reorder sanity (joint multi-task == single-task): **passed**.

Accuracy mean±std (%); chance in parentheses:

| Benchmark | Chance | 30M baseline | 30M Full | 30M Block | 90M baseline | 90M Full | 90M Block |
|---|---:|---:|---:|---:|---:|---:|---:|
| HellaSwag | 25 | 25.3±0.1 | 25.7±0.2 | 25.6±0.5 | 26.6±0.4 | 27.1±0.3 | 27.0±0.0 |
| LAMBADA | — | 0.4±0.1 | 1.2±0.5 | 1.2±0.0 | 12.5±0.7 | 15.4±0.1 | 15.0±0.0 |
| PIQA | 50 | 52.7±0.5 | 53.0±0.2 | 52.6±0.2 | 56.3±0.0 | 57.0±0.1 | 57.3±0.1 |
| WinoGrande | 50 | 49.2±0.9 | 50.2±0.5 | 50.3±1.3 | 50.7±2.3 | 51.3±0.8 | 50.8±0.3 |
| ARC-Easy | 25 | 31.8±0.3 | 31.6±0.1 | 32.2±0.6 | 37.7±0.2 | 38.1±0.8 | 39.0±0.4 |
| ARC-Challenge | 25 | 20.9±0.1 | 21.2±0.5 | 21.2±0.1 | 21.9±0.2 | 22.5±0.2 | 22.5±2.4 |
| OpenBookQA | 25 | 26.7±1.3 | 24.8±0.9 | 24.9±0.8 | 28.3±0.1 | 27.8±0.3 | 28.2±0.0 |
| BoolQ | 50 | 56.1±3.4 | 51.5±8.7 | 43.6±2.7 | 57.8±0.1 | 50.7±1.8 | 48.3±2.8 |
| SciQ | 25 | 34.4±0.3 | 41.8±2.8 | 39.1±3.8 | 65.4±0.8 | 68.1±3.5 | 69.4±0.4 |

At 90M, AttnRes helps most on LAMBADA and SciQ; ARC-Challenge stays below chance for all variants; BoolQ favors baseline. Full artifact: `outputs/benchmark_panel/blessed_panel_latest.{json,csv,md}`.

## Key methodological notes
- All models dense decoder-only; original AttnRes is entirely MoE. Dense reproduction = generalization contribution.
- Ladder holds d/L ~73 constant (matches Moonshot Table 2). Small off-curve models (30M, 90M) shaped for meaningful blocks, not on scaling curve.
- Original used ~8 blocks approximately (per-model counts unspecified, layer counts 12-17 don't all divide by 8); we fix block counts precisely per size for controlled comparison.

### FINDING: Layer-1 magnitude spike in Full AttnRes (90M, dense)

Full AttnRes at 90M develops an extreme hidden-state magnitude spike at layer 1 (||h_1|| ~ 200-260, vs <55 at all other depths). Investigation:

- Develops DURING training (starts ~4.5 at step 100, climbs after ~1.5-2k steps), not at init.
- Cause: layer 1's depth-attention concentrates ~98% of weight on the layer-0 attention output, whose raw magnitude itself grows to ~267.
- Mechanism: AttnRes applies RMSNorm to KEYS (preventing magnitude from biasing attention weights) but sums RAW VALUES. So a heavily-weighted source can have unbounded magnitude.
- Depth-dependence: layer 1 has only 3 sources (embedding, L0 attn out, L0 mlp out), permitting extreme concentration. Deeper layers have more sources; max weight falls to 0.11-0.22.
- NOT present at 30M (final ||h_1|| ~3-8).
- Benign for loss: corr(||h_1||, val_loss) = -0.75 (loss falls as spike grows). Full still beats baseline.

HYPOTHESIS (to test at larger scales): Full's early-layer magnitude explosion is a mild pathology that Block AttnRes avoids via its block structure, explaining why Block outperforms Full at 90M (contrary to the original's reported ordering, where Full is the upper bound).