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

## Phase 0 decisions
- OmegaConf+YAML over Hydra (lighter); Streamlit dashboard (later); MLflow default, W&B behind flag + env key.
