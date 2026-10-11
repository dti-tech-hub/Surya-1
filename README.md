# Adaptive Domain-Aware Representation Learning for Explainable TSAD

B.Tech final-year project. See `docs/DESIGN.md` and `PROGRESS.md`.

## Quick start (Phase 0)
```
pip install -r requirements-smoke.txt
python -m pytest -q
python scripts/smoke.py --config configs/smoke.yaml
```
Full deps: `pip install -r requirements.txt`.
W&B: `set WANDB_API_KEY=...` then `--opt tracking.backend=wandb`. Never commit keys.
