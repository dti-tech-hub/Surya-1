# PROGRESS.md — phase tracker (updated after every phase)

## Status
- [x] Phase 0 — scaffold (2026-10-11)
- [x] Phase 1 — data pipeline (2026-10-11)
- [ ] Phase 2 — eval + baselines
- [ ] Phase 3 — contrastive repr (O1)
- [ ] Phase 4 — adaptive alignment (O2+O3)
- [ ] Phase 5 — explainability (O4)
- [ ] Phase 6 — experiments E1–E8
- [ ] Phase 7 — API + dashboard
- [ ] Phase 8 — reproducibility/docs
- [ ] Phase 9 — Colab/Kaggle
- [ ] Phase 10 — report generator
- [ ] Phase 11 — final audit

## Phase 1 log
- 10/10 tests pass; data_summary smoke wrote reports/data_summary.{csv,md,png} (2 synthetic domains, CPU seconds).
- SMD/SMAP loaders attempted, gracefully skipped (no local files) with download instructions.
- Leakage discipline verified by tests (series-level split, fit-train-only norm, window-disjoint assert).

## Phase 0 log
- Scaffold: pyproject/requirements(-smoke)/Makefile/pre-commit/gitignore, OmegaConf config + CLI overrides,
  seed/logging/MLflow-W&B-off tracker, smoke script + stub entry points, 3 tests.
- Acceptance: `make setup && make test` + `make smoke` on CPU.
