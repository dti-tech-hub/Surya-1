# DESIGN.md — living design doc (Phase 0)

## Architecture
Config-driven TSAD pipeline: data → contrastive encoder → adaptive alignment → prototype scoring → explain.

```mermaid
flowchart LR
  D[Data pipeline] --> E[Encoder + contrast] --> A[Adaptive alignment] --> S[Prototype scoring] --> X[Explain layer]
```

## Tensor convention
All series tensors are `(B, T, C)` = batch, time, channels. Windows sliced from `(T, C)` series.

## Losses (to be fully specified in Phase 3/4)
- `L = L_contrast + α_t·Σ_k w_k·L_align(k, gate) + β·L_preserve + γ·L_sparse`

## Threats to validity (preview)
- Point-adjusted metrics inflate scores → always report strict + PA side by side.
- Thresholds must never use test labels (val-only).
- Normalization stats fit on train only.

## Phase 6 — experiments E1–E8 (implemented)
- common: seed_cfg/ensure_defaults, save_all (CSV+MD+booktabs TEX), paired t/Wilcoxon, mean±std aggregation,
  barplot/heatmap. Dispatcher: --exp/--seeds/--all-smoke/--all.
- E1 baselines-vs-contrast; E2 pairs + gain vs target-only (source-val threshold, zero target labels);
  E3 none/MMD/CORAL/DANN/adaptive (+CORAL/DANN in full mode); E4 kinds{clean,shift,noise} + dissimilarity axis;
  E5 encoder + gate/weights/preserve/monitor toggles (+window/k in full); E6 gallery aggregation;
  E7 alpha×tau grid; E8 params/forward-ms/score-throughput (fit timed separately).
- Smoke = pipeline proof (1 seed, stats n/a); paper claims need 5-seed full runs (E4 seed-dependence documented).

## Phase 5 — explainability O4 (implemented)
- JSON explanation: prototype nearest/medoid + overlay; attention (B,T) w/ heatmap, unavailable-flag for TCN;
  native saliency/IG(20 steps)/smoothgrad + lazy captum/SHAP/LIME backends; GD counterfactual (sparse, bounded,
  plausible train-min/max, validity/sparsity/proximity); cross-domain (per-domain proto dists, gate dims);
  rule-based NL summary (no LLM).
- Metrics: deletion-AUC/sufficiency/comprehensiveness, rank-corr + top-k Jaccard stability, P@k/AUPRC
  localization vs GT masks, CF validity/sparsity/proximity; explainer comparison table in gallery.
- Score fn for grads: softmin prototype distance (differentiable); baseline = medoid normal window.

## Phase 4 — adaptive alignment O2+O3 (implemented, core novelty)
- Losses: MMD (median-heuristic bandwidths — fixed sigmas saturate in high-dim input space), CORAL, DANN
  (GRL + Ganin schedule). Combined: L = Lc + a_t*S w_k*L_align + b*Lp + g*Lsparse, warm-up then monitor-adapted.
- A gate g(z)E[0,1]^d (sigmoid/hard-concrete, L1+entropy); B scale-free weights softmax(-d/mean(d)/tau), tau=1,
  fit epoch-level on INPUT-space windows (batch rep-space weights were non-discriminative); C normal-only +
  compactness/margin/no-collapse; D monitor halves alpha + rollbacks on val-proxy degradation; E AdaBN TTA baseline.
- Ablations: none / mmd / coral / dann / adaptive_full (+cond chamfer option). Evidence: seed0 default-scale
  uniform-mixed collapses (0.318->0.0) while adaptive reaches 0.444 with related weight 0.67.

## Phase 3 — contrastive repr O1 (implemented)
- Encoders TCN/Transformer(attn exposed (B,T))/ResNet1D, (B,T,C)->rep (B,d); TCN/ResNet use max+mean concat
  (spike preservation). Losses: InfoNCE(tau) + TS2Vec-hierarchical + masked-recon aux, config-weighted.
- Lightning module (cosine, clip=1.0, early-stop, ckpt, 16-mixed on CUDA), labels UNUSED in training;
  prototypes fit on normal train windows (assumed-normal) + kNN/Mahalanobis options; window->timestamp mapping,
  smoothing, val-only calibration + guard band, hybrid emb+recon (when w_mask>0).
- Evidence: trained F1 3x random-encoder control; default-scale AUC 0.919 / event-recall 1.0.

## Phase 2 — eval + baselines (implemented)
- Metrics: strict P/R/F1 + AUC-ROC/PR, PA (flagged inflated), simplified event-wise + Tatbul-style range F1
  (VUS/affiliation deferred — noted limitation). Thresholds: best-F1(val labels)/percentile/POT/fixed, val-only.
- Bootstrap CIs + multi-seed mean±std; transfer_report (gain, negative_transfer bool, gap_to_target_only).
- 8 baselines: zscore/iforest/ocsvm/pca/arima(+numpy AR fallback)/lstm_ae/conv_ae/trans_ae.
  Contrastive no-adapt model lands in Phase 3 (O1).

## Phase 1 — data pipeline (implemented)
- `BaseTSADDataset` → `DomainData`; loaders: synthetic (shift knobs + 6 anomaly types + GT masks),
  SMD (machine), SMAP/MSL (entity), UCR (.npz), SWaT/Yahoo (access-required, graceful `DatasetMissingError`).
- Series-level time-ordered split → window after split (no overlap leakage); per-domain z-score fit-train-only
  (constant-ch std→1, NaN-safe); `DomainRegistry` protocols: single / source-target (unlabeled target-train) /
  LODO / multi-source; weak-label fraction flag (default 0); 9-augment `AugmentPipeline` + pseudo-anomaly util.

## Phase 0 decisions
- OmegaConf+YAML over Hydra (lighter); Streamlit dashboard (later); MLflow default, W&B behind flag + env key.
