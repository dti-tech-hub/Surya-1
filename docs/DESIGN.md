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

## Phase 1 — data pipeline (implemented)
- `BaseTSADDataset` → `DomainData`; loaders: synthetic (shift knobs + 6 anomaly types + GT masks),
  SMD (machine), SMAP/MSL (entity), UCR (.npz), SWaT/Yahoo (access-required, graceful `DatasetMissingError`).
- Series-level time-ordered split → window after split (no overlap leakage); per-domain z-score fit-train-only
  (constant-ch std→1, NaN-safe); `DomainRegistry` protocols: single / source-target (unlabeled target-train) /
  LODO / multi-source; weak-label fraction flag (default 0); 9-augment `AugmentPipeline` + pseudo-anomaly util.

## Phase 0 decisions
- OmegaConf+YAML over Hydra (lighter); Streamlit dashboard (later); MLflow default, W&B behind flag + env key.
