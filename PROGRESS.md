# PROGRESS.md — phase tracker (updated after every phase)

## Status
- [x] Phase 0 — scaffold (2026-10-11)
- [ ] Phase 1 — data pipeline
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

## Phase 0 log
- Scaffold: pyproject/requirements(-smoke)/Makefile/pre-commit/gitignore, OmegaConf config + CLI overrides,
  seed/logging/MLflow-W&B-off tracker, smoke script + stub entry points, 3 tests.
- Acceptance: `make setup && make test` + `make smoke` on CPU.
