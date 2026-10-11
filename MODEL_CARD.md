# MODEL_CARD.md — synthetic_tcn bundle (smoke demo)

- **What:** TCN contrastive encoder (d=8) + k-means prototypes (k=3), trained on synthetic domain_0
  (length 1024, 2ch, seed 0, 2 epochs, InfoNCE). Bundle: `models/bundle/synthetic_tcn/` (rebuilt, git-ignored).
- **Intended use:** Demo/API smoke tests only — NOT a production detector.
- **Measured:** strict_F1 0.074, AUC-ROC 0.83, event-recall 1.0, precision ≈0.07 (over-flags; see reports/).
- **Limitations:** Point-spike dilution; unsupervised threshold conservative; synthetic-only (no real-patient data).
- **Reproduce:** `python scripts/build_bundle.py --config configs/smoke.yaml --name synthetic_tcn`.
- **Ethics:** No personal data; synthetic series. Real deployments need domain validation + human review of flags.
