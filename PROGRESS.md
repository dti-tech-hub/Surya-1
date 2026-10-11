# PROGRESS.md — phase tracker (updated after every phase)

## Status
- [x] Phase 0 — scaffold (2026-10-11)
- [x] Phase 1 — data pipeline (2026-10-11)
- [x] Phase 2 — eval + baselines (2026-10-11)
- [x] Phase 3 — contrastive repr O1 (2026-10-11)
- [x] Phase 4 — adaptive alignment O2+O3 (2026-10-11)
- [ ] Phase 5 — explainability (O4)
- [ ] Phase 6 — experiments E1–E8
- [ ] Phase 7 — API + dashboard
- [ ] Phase 8 — reproducibility/docs
- [ ] Phase 9 — Colab/Kaggle
- [ ] Phase 10 — report generator
- [ ] Phase 11 — final audit

## Phase 4 log
- 32/32 tests pass. MMD(median-heuristic)/CORAL/DANN+GRL+schedule; gate(sigmoid/hard-concrete+sparsity);
  scale-free domain weights; preserve(compactness+margin+no-collapse); monitor(halve-alpha+rollback); AdaBN TTA.
- Smoke cmd: `python scripts/adapt.py --config configs/smoke.yaml --epochs 5` -> reports/adapt_smoke.{csv,md}, adapt_mmd.png:
  source-only 0.065 / uniform-clean 0.0 / uniform-mixed 0.0 (NEG-TRANSFER True) / adaptive-full 0.25 (weights [0.78,0.22]).
- Default-scale seed0: source-only 0.0 / uniform-clean 0.318 / uniform-mixed 0.0 / adaptive 0.444 (best).
  Seed1: all 0.18-0.31, adaptive joint-best, never worse. Claim = robustness, not dominance (E4: 5 seeds).
- Key fixes: per-split anomaly injection; noise-harmful domain (shift erased by per-domain norm); best-F1-on-VAL
  operating point for fair transfer comparison (test labels never used).

## Phase 3 log
- 25/25 tests pass. Contrastive TCN (InfoNCE, Lightning, cosine, clip, early-stop, ckpt) + prototype/kNN/Mahalanobis
  scoring + hybrid-recon option + t-SNE/UMAP plot saved (reports/embed_tcn_domain_0.png).
- REAL numbers: default-scale strict_f1=0.229, AUC-ROC=0.919, event_recall=1.0 (finds all events, over-flags ->
  precision 0.13); trained 3x F1 over random-encoder control (0.069). Smoke (tiny, 2ep): F1=0.074, AUC=0.83.
- Honest limits: mean-pool dilution of point spikes (fixed partly via max+mean concat); unsupervised-percentile
  operating point is conservative vs ranking quality (E7 will study thresholds); val-only calibration + guard band added.

## Phase 2 log
- 20/20 tests pass; `scripts/eval.py` smoke: 8 baselines x 2 synthetic domains = 16 rows -> reports/baselines_smoke.{csv,md}.
- Honest smoke finding: 1-epoch AEs score strict_f1~0 on tiny CPU data while pa_f1 hits 1.0 — PA inflation visible, as warned.
- Thresholds fit on VAL only; transfer_report() flags negative_transfer when gain<0.

## Phase 1 log
- 10/10 tests pass; data_summary smoke wrote reports/data_summary.{csv,md,png} (2 synthetic domains, CPU seconds).
- SMD/SMAP loaders attempted, gracefully skipped (no local files) with download instructions.
- Leakage discipline verified by tests (series-level split, fit-train-only norm, window-disjoint assert).

## Phase 0 log
- Scaffold: pyproject/requirements(-smoke)/Makefile/pre-commit/gitignore, OmegaConf config + CLI overrides,
  seed/logging/MLflow-W&B-off tracker, smoke script + stub entry points, 3 tests.
- Acceptance: `make setup && make test` + `make smoke` on CPU.
