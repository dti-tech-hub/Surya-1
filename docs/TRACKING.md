# Experiment tracking

- Default backend: MLflow (`tracking.backend: mlflow`, `tracking.experiment: tsad-default`).
  `train.py` logs params + metrics. View: `mlflow ui` (needs `pip install mlflow` — full requirements).
- Smoke default is `off` (no server, no deps) — CI-friendly.
- W&B (optional): `set WANDB_API_KEY=...` (never commit it), then
  `python scripts/train.py --config configs/default.yaml --opt tracking.backend=wandb`.
- Every run also saves `reports/last_train_config.yaml` (config + seed) for traceability;
  git hash capture lands with the Phase 10 report generator.
