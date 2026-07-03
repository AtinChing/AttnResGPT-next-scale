# AttnRes Reproduction: Reference Notes

Canonical reference sheet for the reproduction-and-extension study of Attention Residuals (AttnRes), Kimi Team / Moonshot AI, arXiv 2603.15031 (16 Mar 2026).

Target: TMLR with Reproducibility Certification, then NeurIPS MLRC 2026.

This is a living document. Keep it version-controlled in the repo. It holds: the original paper's claims and results, our reproduction plan, our decisions and their justifications, and known differences from the original.

---

## 1. What the original paper is

AttnRes replaces standard PreNorm residual connections (which add every prior layer output with fixed unit weight) with a learned, input-dependent softmax attention over prior layer outputs. Each layer has a learned pseudo-query that decides how much to pull from each earlier source. The motivation: standard residuals cause hidden-state magnitude to grow with depth (PreNorm dilution), burying early-layer information.

Two variants:
- **Full AttnRes**: each layer attends over ALL prior sublayer outputs. Cost O(L^2) compute, O(Ld) memory.
- **Block AttnRes**: layers grouped into N blocks. Standard additive residuals inside a block; softmax depth-attention only over block-level summaries (plus the current partial sum). Cost reduced to O(N^2) / O(Nd). This is the scalable variant and the one tied to the stabilization and compute claims.

---

## 2. The three training efforts (do not confuse them)

### (A) Scaling law sweep  [Table 2, Figure 4]
- Five model sizes: 194M, 241M, 296M, 436M, 528M ACTIVATED params.
- These are MoE (Mixture of Experts) models. "Activated params" = params used per token (MoE routes each token through a subset). Total params NOT reported for the sweep; only activated counts given.
- Three variants trained per size: baseline (PreNorm), Full AttnRes, Block AttnRes (~8 blocks). Plus mHC-lite as a reference competitor.
- Trained on 38B to 119B tokens each, context length 8192.
- Reported: final validation loss per variant per size, plus configs (params, tokens, layers L_b, heads H, d_model, d_ff, lr, batch size).
- Fitted power-law curves L = A * C^(-alpha), C = compute in PFLOP/s-days:
  - Baseline:  L = 1.891 * C^(-0.057)
  - Full:      L = 1.865 * C^(-0.057)
  - Block:     L = 1.870 * C^(-0.058)

### (B) Main production model  [Table 3, Figure 5]
- One big model: full Kimi Linear config, 48B total / 3B activated params.
- 27 transformer blocks (54 layers), Block AttnRes with 6 layers per block -> 9 blocks + embedding = 10 depth-wise sources.
- Trained on 1.4T tokens (1T pretraining + ~400B mid-training), 4096 context, Muon optimizer, WSD schedule.
- Compared against an identically-trained baseline 48B.

### (C) Ablation / analysis models  [Table 4, Figures 6, 7, 8]
- Smaller 16-layer models used to justify design choices and analyze behavior.

---

## 3. The claims, mapped to evidence and to our reproduction

| # | Claim | Where proven in original | Our reproduction route | Feasible at our scale? |
|---|-------|--------------------------|------------------------|------------------------|
| 1 | Quality gain: AttnRes beats baseline on loss | Sweep + 48B | Our scaling ladder, baseline vs AttnRes vs Block | Yes |
| 2 | Mechanism: learned input-dependent softmax over depth | Section 3, Eq 2-4 | Implemented in our Full + Block | Yes |
| 3 | Bounded magnitude / stabilization | 48B, Figure 5b | Per-layer activation-norm trajectory plots, baseline vs Full vs Block | Yes, and refined |
| 4 | More uniform gradient distribution across depth | 48B, Figure 5c | Per-layer gradient-norm plots, baseline vs Full vs Block | Yes |
| 5 | 1.25x compute advantage (flagship, BLOCK) | Sweep, fitted curves | Iso-loss curve fit across our ladder; measure horizontal offset | Yes, but see scrutiny notes |
| 6 | Downstream task transfer | 48B, Table 3 | Scale-appropriate benchmark subset on our trained models | Partial (scale-limited) |
| 7 | Block ~8 recovers most of Full's benefit | Ablation, Figure 6 | Block-size sweep at small scale | Yes, cheap |
| 8 | Architecture shift toward deeper-narrower | Figure 7 | Depth/width sweep at fixed compute | Yes, cheap |

**Refinement to claims 3/4 (measured early, pre-training):** The original frames stabilization as a Block property (periodic reset, Figure 5b sawtooth). Our measured per-layer residual-stream norms show a more complete picture: baseline grows monotonically with depth (0.225 to 0.372, the PreNorm dilution), while BOTH Full and Block bound this growth, via different mechanisms. Full bounds it flattest via convex combination (softmax weights sum to 1; measured 0.091 to 0.062). Block bounds it in a sawtooth (grows within a block, drops at each reset). So stabilization is not exclusively a Block property; it is a property of both variants through distinct mechanisms. This is a more precise account than the original's Block-centric framing and is reproduction added value. Gradient-distribution (claim 4) still to be measured during training.


Component ablations (Table 4), cheap and high-value to reproduce:
- input-independent mixing hurts (1.749 vs 1.737 Full)
- sigmoid instead of softmax hurts (1.741)
- removing RMSNorm on keys hurts (1.743 Full / 1.750 Block)
- multihead depth-attention hurts (1.752 vs 1.746 Block)
- pseudo-query MUST be zero-initialized (prevents training volatility)

---

## 4. Original downstream results  [Table 3, 48B model]

Baseline -> AttnRes:
- General: MMLU 73.5 -> 74.6, MMLU-Pro 52.2 -> 52.2, GPQA-Diamond 36.9 -> 44.4, BBH 76.3 -> 78.0, ARC-Challenge 64.6 -> 65.7, HellaSwag 83.2 -> 83.4, TriviaQA 69.9 -> 71.8
- Math & Code: GSM8K 81.7 -> 82.4, MGSM 64.9 -> 66.1, Math 53.5 -> 57.1, CMath 84.7 -> 85.1, HumanEval 59.1 -> 62.2, MBPP 72.0 -> 73.9
- Chinese: CMMLU 82.0 -> 82.9, C-Eval 79.6 -> 82.5

Largest gains on reasoning (GPQA +7.5, Math +3.6, HumanEval +3.1). Note: these are at 48B scale; most will be at or near chance at our scale, so we reproduce only the subset that shows above-chance signal and honestly scope out the rest.

---

## 5. The 1.25x claim: how it was derived and how we scrutinize it

How they got it: NOT from a single point. They fit two power-law curves across the five compute budgets, then read the horizontal (compute) distance between the baseline and Block curves at a fixed loss. At ~5.6 PFLOP/s-days, Block reaches loss 1.692 vs baseline 1.714, expressed as "Block matches a baseline trained with 1.25x more compute."

Legitimate critiques to investigate rigorously (these are the REAL weaknesses, not "they used one point"):
- The two fitted curves are extremely close: exponents -0.057 vs -0.058, intercepts 1.891 vs 1.870 (~1% apart). A compute-multiplier extracted from near-parallel curves is highly sensitive to fit error.
- Only five points per curve is a thin basis for fitting a power law and then extracting a ratio between two such fits. Likely wide uncertainty; they report no confidence interval on 1.25x.
- The advantage is not constant across scale (Full-vs-Block gap shrinks to 0.001 at the largest size). So 1.25x is a point-estimate at one compute level, not a universal constant.

Our contribution here: reproduce the fit at our scale, quantify the uncertainty on the multiplier (bootstrap / vary fit range), and report how fragile or robust 1.25x is. A well-characterized uncertainty on the flagship number is genuine added value.

---

## 6. Known differences from the original (state these explicitly in the paper)

- **MoE vs dense**: original sweep uses MoE (Kimi Linear); we use dense decoder-only. Framing: we test whether AttnRes generalizes to dense decoder-only models, a widely-used class, which also isolates the residual mechanism from MoE routing effects. Scoped difference, not a hidden one.
- **Scale**: original sweep 194M-528M activated on 38-119B tokens; 48B on 1.4T tokens. Ours: ~30M-350M (up to ~700M on Colab) dense, far fewer tokens. Claims stay relative (baseline vs AttnRes at matched budget). Some effects (per original) amplify with scale, so our magnitudes may differ; stated as a limitation.
- **Optimizer / schedule**: original uses Muon + WSD. Note whatever we actually use (e.g. AdamW + cosine) and that hyperparameters are held identical across variants within each comparison for fairness.
- **Corpus**: original uses the Kimi Linear 1.4T corpus. Ours: a naturalistic corpus slice (FineWeb / C4 / Pile, TBD) for the backbone so benchmarks show above-chance signal; TinyStories only as an optional small supplementary/robustness note, not the backbone.

---

## 7. Our execution plan (backbone + extensions)

1. **Block AttnRes implementation** (gating build). Faithful to paper: mixer at every sublayer over block-level sources + partial sum, additive accumulation inside blocks, reset at block boundary. Zero-init query, RMSNorm on keys. Verified against Figure 2 pseudocode and Eq 5-6.
2. **Iterate on RTX 3070** to lock config, hyperparameters, LR schedule, block count, and the minimum token budget on the naturalistic corpus that gets baseline above chance on 2-3 easy benchmarks (HellaSwag, LAMBADA, ARC-Easy).
3. **Bless the config**, then spend Colab T4 units on the real backbone: baseline vs Full vs Block across the size ladder (up to ~700M), naturalistic corpus, multi-seed.
4. **Iso-loss compute sweep + overhead measurement** -> reproduce claim 5, with uncertainty quantification.
5. **Per-layer norm + gradient trajectory analysis** (Full and Block) -> reproduce claims 3, 4.
6. **Scale-appropriate downstream benchmark subset** -> reproduce claim 6, honestly scoped.
7. **Cheap extensions**: block-size sweep (claim 7), architecture-shift depth/width sweep (claim 8), component ablations (Table 4).
8. **Document every discrepancy** between Moonshot's paper and their released code (proven certification fuel).

NOT doing: distillation (adds a confound to a controlled architecture comparison), encoder architectures for v1 (generalization, holds for later / scope control), VLM (that is Paper 2).

---

## 8. Glossary of terms used

- **Blessed config**: the one final, locked configuration committed to real compute after laptop iteration. Canonical, goes in the paper.
- **Activated params (MoE)**: params actually used per token in a Mixture-of-Experts model (only a subset of total params route each token).
- **Iso-loss / compute-equivalence**: the horizontal distance between two loss-vs-compute curves at a fixed loss; how much more compute one variant needs to match another's quality.
- **PFLOP/s-days**: compute unit; sustained PFLOP/s for one day. The x-axis of the scaling curves.
- **PreNorm dilution**: with PreNorm residuals, hidden-state magnitude grows ~O(L) with depth, so each layer's relative contribution shrinks; early-layer info gets buried.
- **Block reset**: at a block boundary, the intra-block partial sum is finalized and the accumulator resets; this is what bounds magnitude growth (the stabilization mechanism).
- **Pseudo-query**: the single learned d-dimensional vector per layer that computes the depth-attention weights. Must be zero-initialized.

---

## 9. Venue facts (for reference)

- TMLR reproducibility certification: "papers whose primary purpose is reproduction of other published work and that contribute significant added value through additional baselines, analysis, ablations, or insights."
- MLRC 2026: "any paper or set of papers" (single-paper is eligible). Emphasis on reproducibility, replicability, generalizability, and stress-testing. Negative/partial results explicitly welcomed.
- Pipeline: TMLR acceptance + reproducibility certification -> light MLRC compatibility review -> presented at NeurIPS 2026 (Sydney).
- Hard deadline: TMLR decision in the MLRC system by Sept 30 2026. Aim to submit to TMLR by end of June / early July given ~3-month review.