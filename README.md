# Adaptive Domain-Aware Representation Learning for Explainable Time Series Anomaly Detection

B.Tech final-year project (4 students): self-supervised contrastive representations + adaptive
cross-domain alignment with negative-transfer guards + a 6-part explainability layer.

## Install

```bash
pip install -r requirements-smoke.txt   # scaffold+tests (seconds)
pip install -r requirements.txt         # full stack (torch CPU, lightning, API, dashboard)
# GPU box / Colab: plain `pip install -r requirements.txt` picks CUDA torch automatically.
```

## Quick start (CPU, ~3 min)

```bash
python -m pytest -q                                   # 52 tests
python scripts/data_summary.py --config configs/smoke.yaml      # reports/data_summary.{csv,md,png}
python scripts/eval.py --config configs/smoke.yaml --out reports/baselines_smoke
python scripts/train.py --config configs/smoke.yaml --out reports/contrast_smoke
python scripts/adapt.py --config configs/smoke.yaml --epochs 5 --out reports/adapt_smoke
python scripts/make_explanations.py --config configs/smoke.yaml --out reports/explain_smoke
python scripts/run_experiment.py --config configs/smoke.yaml --all-smoke --outdir reports/exp
python scripts/build_bundle.py --config configs/smoke.yaml --name synthetic_tcn
python -m uvicorn tsad.api.app:app --app-dir src --port 8000  # make app
python -m streamlit run dashboard/app_streamlit.py            # make dashboard
docker compose up   # needs Docker; bakes bundle at build, serves api:8000 + dashboard:8501
```

Makefile shortcuts: `make setup test lint smoke train eval explain app bundle report experiments-smoke`.

## Data setup (real datasets)

| Dataset | Access | Destination |
|---|---|---|
| SMD (28 machines, 38ch) | public ([OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly)) | `data/smd/machine-<i>-{train,test,test_label}.npy` |
| SMAP/MSL | public mirror ([telemanom](https://github.com/khundrakpam/telemanom)) | `data/smap_msl/{train,test,test_label}/*.npy` |
| UCR Anomaly Archive | public ([UCR](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)) | `data/ucr/<group>__<name>.npz` (`series`, `labels`) |
| SWaT | **request** ([iTrust](https://itrust.sutd.edu.sg/itrust-labs_datasets/)) | `data/swat/` |
| Yahoo S5 | **request** (Webscope L6) | `data/yahoo/` |

Missing files → loaders raise `DatasetMissingError` with the URL (never fake data). Without downloads,
everything runs on the synthetic generator (shift knobs + 6 anomaly types + GT channel masks).
`python scripts/download_notes.py` prints all instructions. Details: `docs/DATA_SETUP.md`.

## Reproduce each experiment

```bash
python scripts/run_experiment.py --config configs/smoke.yaml --exp e1 --outdir reports/exp  # ... e1..e8
python scripts/run_experiment.py --config configs/smoke.yaml --all-smoke --outdir reports/exp
python scripts/run_experiment.py --config configs/default.yaml --all --outdir reports/exp_full  # 5 seeds, GPU box
```

## Results summary (actual runs, 2026-10-11; see `reports/`)

- Single-domain: classicals lead on point spikes (iforest strict_F1≈0.61 smoke); contrastive-TCN finds
  **all events (recall 1.0, AUC 0.92)** but over-flags (precision ≈0.13) under the unsupervised threshold.
- Transfer (default-scale, seed 0): uniform-mixed collapses 0.318→0.000 with a noise source;
  **adaptive-full reaches 0.444**, related-source weight 0.67. Seed 1: parity (adaptive joint-best).
- Explanations: IG deletion-AUC 0.5–0.8, localization P@k 1.0 on clean hits, counterfactual validity 10/10.
- Full numbers: `reports/*.{csv,md}`, `reports/exp/`, `reports/explain_smoke/`. `make report` (Phase 10)
  turns these into paper assets; anything marked `NOT RUN` was not run — we do not fabricate SOTA claims.

## Limitations (honest)

1. Mean-pooled window embeddings dilute isolated point spikes (mitigated: max+mean concat).
2. Unsupervised-percentile operating point is conservative vs ranking quality (E7 studies thresholds).
3. Negative-transfer effect is seed-dependent on tiny data; 5-seed full runs required for paper claims.
4. Point-adjusted metrics reported alongside strict only — PA inflates, never trust it alone.
5. VUS/affiliation metrics deferred; range/event-wise F1 implemented (simplified Tatbul).

## Repo map

`src/tsad/{data,eval,baselines,models,losses,alignment,lightning,explain,api,viz}` ·
`scripts/{train,eval,adapt,make_explanations,run_experiment,build_bundle,data_summary,make_report}` ·
`experiments/{common,e1..e8}` · `configs/{default,smoke}.yaml + {dataset,model,align}/` ·
`docs/{DESIGN,VIVA_QA,DATA_SETUP,CONFERENCE}` (CONFERENCE lands Phase 10) · `PROGRESS.md`.
