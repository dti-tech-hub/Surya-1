# GPU runs (Colab / Kaggle) — runtimes, memory, resume

## How to run
- Colab: open `notebooks/run_on_colab.ipynb`, set `REPO_URL` (or upload `repo.zip`), Runtime → T4 GPU.
- Kaggle: `notebooks/run_on_kaggle.ipynb`, attach repo dataset/zip, GPU on, Internet on for pip.
- Default notebook plan: `configs/fast.yaml` (2000-pt, 3 epochs) × seeds 0–2, E1–E8.
- Paper plan: `SEEDS='0 1 2 3 4'` + `configs/default.yaml` (5 epochs).

## Runtimes

`measured` = wall-clock on the dev CPU box (this repo, Oct 2026). `~T4` = projection (torch ops 5–10×
faster; sklearn/ARIMA stay CPU-bound). Projections are estimates, not claims.

| Task | Config | Measured CPU | ~T4 GPU |
|---|---|---|---|
| pytest (52) | smoke | ~35 s | n/a (CPU) |
| all-smoke E1–E8 | smoke.yaml, 1 seed | ~2.5 min | ~1.5 min |
| train (contrast) | default, 5 ep | ~50 s | ~15 s |
| adapt (4 conds) | default, 5 ep | ~45 s | ~15 s |
| gallery (10) | smoke | ~50 s | ~20 s |
| E1 full (8 bl + contrast) | fast, 3 seeds | ~10 min est | ~6 min est (ARIMA CPU-bound) |
| E2/E3/E4/E5/E7 | fast, 3 seeds each | ~8–15 min est each | ~3–5 min est each |
| E6 (3 galleries) | fast | ~3 min est | ~1.5 min est |
| **E1–E8 total** | fast, 3 seeds | **~60–75 min est** | **~20–30 min est** |
| E1–E8 paper | default 5ep, 5 seeds | hours (CPU) | ~1 h est |

## Memory (estimates, conservative)
- CPU RAM < 4 GB (windows ≤ ~200 × 64 × 4 floats; batch 64). VRAM < 2 GB — any T4 (16 GB) is ample.
- If OOM: halve `train.batch_size` (CLI `--opt train.batch_size=32`), or use `configs/smoke.yaml`.

## Timeouts & resume
- Colab: ~90 min idle / ~12 h max; Kaggle: ~9 h run, 20 h GPU/week.
- Resume is experiment-granular: rerun the loop cell, finished exps skip via `reports/exp/<e>.csv`.
- `train.py` keeps best `checkpoints/` via Lightning `ModelCheckpoint` (see cell notes for manual resume).
- Always `cp -r reports (+ models)` to Drive/working after each exp (notebook does this).
