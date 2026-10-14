# Surya-1 · Adaptive Domain-Aware Representation Learning for Explainable Time Series Anomaly Detection

Self-supervised contrastive representations → adaptive cross-domain alignment with negative-transfer
guards → a six-part explainability layer. CPU-first, fully reproducible, no fabricated numbers.

**Team:** B.Tech final-year project (4 students) · **Status:** 12/12 phases complete · **Tests:** 61 passing ·
**Audit:** 15/15 checks PASS

---

## 1. The problem

Deploying an anomaly detector on a new machine/site usually fails in one of two ways:

1. **Distribution shift** — the new domain's normal behaviour differs, so the model flags everything
   (or nothing). Naively mixing source domains can be *worse* than using none.
2. **No explanation** — an operator gets "anomaly at t=412" with no reason, so the alert is unusable.

This project attacks both: learn representations that transfer, adapt them safely, and explain every flag.

## 2. Contributions

| ID | Objective | Where | Evidence |
|---|---|---|---|
| **O1** | Self-supervised contrastive representations (InfoNCE / TS2Vec / masked-recon; TCN, Transformer, ResNet encoders) | `src/tsad/{models,losses,lightning/contrast.py}` | E1, `reports/contrast_default.md` |
| **O2** | Adaptive cross-domain alignment (MMD, CORAL, DANN) with a learned feature gate + domain weights | `src/tsad/alignment/`, `src/tsad/lightning/adapt.py` | E3, `reports/adapt_default.md` |
| **O3** | Negative-transfer guard (anomaly-preservation constraint + monitor; a transfer is *rejected* if it hurts) | `src/tsad/alignment/transfer_monitor.py` | E4, `reports/adapt_default.md` |
| **O4** | Six-part explanations: prototype/medoid, attention, attribution (IG/Saliency/SmoothGrad/KernelSHAP), counterfactual, cross-domain verdict, natural-language summary | `src/tsad/explain/` | E6, `reports/explain_smoke/` |

## 3. Real results

All numbers below are **actual outputs** committed under `reports/` (regenerate with the commands in §5).
Metrics are **strict** (point-wise); point-adjusted (PA) numbers are reported alongside everywhere because PA
inflates scores — never quote PA alone.

### 3.1 Single-domain detection — `reports/contrast_default.md`

| model | strict_F1 | precision | recall | AUC-ROC | AUC-PR | event_F1 | range_F1 |
|---|---|---|---|---|---|---|---|
| **contrast-TCN (Ours)** | 0.2288 | 0.1292 | **1.0000** | **0.9192** | **0.4484** | **1.0000** | **0.7217** |

> The contrastive model **finds every injected event** (recall 1.00, event-F1 1.00) but over-flags under the
> unsupervised percentile threshold (precision 0.13). This is a *thresholding* problem, not a representation
> problem — ranking quality is strong. E7 studies exactly this.

### 3.2 Transfer — `reports/adapt_default.md` (2 domains, seed 0)

| condition | domain weights | strict_F1 | AUC-ROC |
|---|---|---|---|
| source-only | [1.0, 1.0] | 0.0000 | 0.9317 |
| uniform mix (clean source) | [1.0] | 0.3176 | 0.9478 |
| **uniform mix (noisy source)** | [0.5, 0.5] | **0.0000** | 0.9307 |
| **adaptive-full (ours)** | **[0.674, 0.326]** | **0.4444** | 0.9225 |

> **The headline result (seed 0):** mixing a noisy source uniformly destroys performance
> (0.3176 → 0.0000). Adaptive weighting down-weights the noisy source to 0.326 and reaches **0.4444** —
> the best of the four conditions, with no negative-transfer flag raised.
>
> **Seed 1 does not reproduce the collapse** (`reports/adapt_default_s1.md`): there, uniform-mixed scores
> 0.3019 and adaptive-full 0.3059 — adaptive is still (marginally) best, but by only +0.004. The
> *collapse* phenomenon is therefore **seed-0-specific on this small 2-domain dataset**, and the
> adaptive-over-uniform margin ranges from +0.004 to +0.444 depending on seed. This is exactly why §7
> lists multi-seed runs as a prerequisite for any paper claim.

### 3.3 Baselines — `reports/baselines_smoke.md` (domain_0, strict_F1)

z-score **0.7273** · ARIMA **0.7273** · IsolationForest 0.7143 · OCSVM 0.6667 · LSTM-AE 0.5455 ·
Conv-AE 0.5000 · Transformer-AE 0.5714 · PCA 0.3077

> Classical methods win on *point spikes* in this synthetic regime. We report this honestly rather than
> hiding it: the contribution is transfer robustness + explanations, not SOTA on single-domain spikes.

### 3.4 Explainability — `reports/explain_smoke/_metrics.md`, `_cf_metrics.md`

Averaged over the **10** inspected anomaly windows (not the best window):

| explainer | deletion-AUC (mean) | deletion-AUC (range) | localisation P@k (mean) | windows with P@k = 1.0 |
|---|---|---|---|---|
| Integrated Gradients | 0.551 | 0.147 – 0.792 | 0.387 | 3/10 |
| Saliency | 0.488 | −0.214 – 0.788 | 0.407 | 3/10 |
| SmoothGrad | 0.520 | −0.299 – 0.791 | 0.423 | 3/10 |

Counterfactual validity: **10/10 windows** — every flagged window admits an edit that returns the score
below threshold.

> **Honest reading:** attribution quality is **highly window-dependent**. On clean, well-localised hits the
> top-ranked channel is the truly injected one (P@k = 1.0) and deletion-AUC reaches ≈0.79, but several
> windows score poorly — including **negative** deletion-AUC, meaning the attribution ranked the responsible
> channel *below* a perturbation baseline. Do not read the per-window best case as the typical case; the
> spread above is the result.

## 4. Architecture

```
src/tsad/
  data/         synthetic generator (shift knobs, 6 anomaly types, GT channel masks) + 5 real loaders
                windows · splits · per-domain normalisation · augmentations
  models/       TCN / Transformer / ResNet encoders · prototype bank · scoring
  losses/       InfoNCE, TS2Vec, masked-recon, supervised-contrastive (weak labels, off by default)
  alignment/    MMD · CORAL · DANN · feature gate · anomaly-preservation · transfer monitor
  lightning/    ContrastModule (O1) · AdaptModule (O2/O3)
  explain/      prototype · attention · attribution · counterfactual · cross-domain · NL summary
  eval/         strict + PA + range/event metrics · thresholds · bootstrap CIs
  baselines/    z-score · ARIMA · iforest · OCSVM · PCA · LSTM-AE · Conv-AE · Trans-AE
  api/          FastAPI service + bundle registry
  viz/          score plots · embedding projections
experiments/    E1–E8 harness (multi-seed, CSV + Markdown + LaTeX, statistics, plots)
scripts/        train · eval · adapt · make_explanations · run_experiment · build_bundle · serve · audit
frontend/       standalone dark-theme dashboard (single HTML file, no build step)
samples/        sample_anomaly.csv (spike at t=50–55) · sample_series.csv
```

**Leakage controls** (audited, `reports/audit.md`): per-domain normalisation is fit on train windows only and
`transform()` asserts `fit()` ran first; labels default to `-1` and enter the loss *only* when
`loss.w_sup > 0` **and** `data.weak_labels > 0` (both default `0`); thresholds are calibrated on validation
labels only; test labels are used for scoring, never fitting.

## 5. Quick start (CPU, ~3 min)

```bash
pip install -r requirements-smoke.txt          # scaffold + tests (seconds)
pip install -r requirements.txt                # full stack (torch CPU, lightning, API, dashboard)

python -m pytest -q                            # 61 tests
python scripts/audit.py --out reports/audit.md # 15/15 checks (leakage, repro, gates, O1–O4)
```

Makefile shortcuts: `make setup test lint smoke train eval explain app bundle report experiments-smoke`.

### Train, evaluate, adapt, explain

```bash
python scripts/data_summary.py    --config configs/smoke.yaml
python scripts/train.py           --config configs/smoke.yaml --out reports/contrast_smoke
python scripts/eval.py            --config configs/smoke.yaml --out reports/baselines_smoke
python scripts/adapt.py           --config configs/smoke.yaml --epochs 5 --out reports/adapt_smoke
python scripts/make_explanations.py --config configs/smoke.yaml --out reports/explain_smoke
python scripts/run_experiment.py  --config configs/smoke.yaml --all-smoke --outdir reports/exp
python scripts/make_report.py     --out reports/final
```

### Serve the API + dashboard

```bash
python scripts/build_bundle.py --config configs/smoke.yaml --name synthetic_tcn
python scripts/serve.py                 # API http://127.0.0.1:8000  (docs at /docs)
python scripts/serve.py --port 8010     # if 8000 is taken
```

Or manually: `python -m uvicorn tsad.api.app:app --app-dir src --port 8000`

**API endpoints:** `GET /health` · `GET /models` · `GET /domains` · `POST /detect` ·
`POST /detect_csv` · `POST /explain` · `POST /adapt` (target-time prototype refresh, unlabeled)

### Web dashboard

Two frontends, both served locally:

| Frontend | How | Notes |
|---|---|---|
| **Standalone web UI** | open `frontend/dashboard.html` in a browser (API must be running) | dark theme, live score chart with flagged-region shading, draggable threshold, click any point for the full 6-part explanation + raw JSON |
| **Streamlit** | `python -m streamlit run dashboard/app_streamlit.py` | simpler, no API required |

`frontend/dashboard.html` talks to the API at `http://127.0.0.1:8000` by default — change the **API base URL**
field in the UI if you used another port. It has a built-in sample series and can also read
`samples/sample_anomaly.csv`.

### Docker

```bash
docker compose up    # api:8000 + dashboard:8501; bakes the bundle at build time
```

## 6. Data setup

| Dataset | Access | Destination |
|---|---|---|
| SMD (28 machines, 38 ch) | public — [OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly) | `data/smd/machine-<i>-{train,test,test_label}.npy` |
| SMAP / MSL | public mirror — [telemanom](https://github.com/khundrakpam/telemanom) | `data/smap_msl/{train,test,test_label}/*.npy` |
| UCR Anomaly Archive | public — [UCR](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) | `data/ucr/<group>__<name>.npz` (`series`, `labels`) |
| SWaT | **request** — [iTrust](https://itrust.sutd.edu.sg/itrust-labs_datasets/) | `data/swat/` |
| Yahoo S5 | **request** — Webscope L6 | `data/yahoo/` |

Missing files raise `DatasetMissingError` **with the source URL** — the code never fabricates data. Without
any downloads everything runs on the built-in synthetic generator. Details: [`docs/DATA_SETUP.md`](docs/DATA_SETUP.md),
and `python scripts/download_notes.py` prints the instructions.

## 7. Reproducibility & honesty

- Every seed produces identical synthetic series, labels, masks and weight initialisation
  (checked in `reports/audit.md` → `repro/*`).
- `make report` / `python scripts/make_report.py` build tables, figures and drafted prose **only** from
  committed numbers. Anything not run is printed as `NOT RUN` — we never fabricate SOTA comparisons.
- Numbers in §3 come from the **default** config on 2 synthetic domains. Real-dataset results are **not yet
  included**; see below.

### Known limitations

1. Mean-pooled window embeddings dilute isolated point spikes (mitigated by max+mean concatenation).
2. The unsupervised percentile operating point is conservative relative to ranking quality.
3. The negative-transfer effect is **seed-dependent on small data** — 5-seed runs are required before any
   paper claim. Only seeds 0 and 1 are committed here.
4. Point-adjusted metrics are always shown next to strict metrics; PA inflates and must not be trusted alone.
5. VUS / affiliation metrics are not implemented; range-wise and event-wise F1 are (simplified Tatbul).
6. Real SMD / SMAP numbers, GPU-scale multi-seed statistics, and conditional-alignment end-to-end runs are
   still outstanding — see `docs/CONFERENCE.md` for the exact remaining plan.
7. DeepSHAP is incompatible with the BatchNorm TCN encoder, so KernelSHAP is used instead.

## 8. Documentation

| File | Contents |
|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | architecture, phase log, design decisions and their rationale |
| [`docs/DATA_SETUP.md`](docs/DATA_SETUP.md) | per-dataset download and directory instructions |
| [`docs/VIVA_QA.md`](docs/VIVA_QA.md) | 40 viva questions with answers |
| [`docs/CONFERENCE.md`](docs/CONFERENCE.md) | narrow evidence-backed claim + remaining work plan |
| [`docs/GPU_RUNS.md`](docs/GPU_RUNS.md) | Colab/Kaggle runtimes, memory, resume procedure |
| [`MODEL_CARD.md`](MODEL_CARD.md) | intended use, limitations, ethics |
| [`PROGRESS.md`](PROGRESS.md) | phase-by-phase build log |
| `reports/audit.md` | generated audit — leakage, reproducibility, gates, objective coverage |

## 9. Reproduce the headline numbers

```bash
python scripts/train.py --config configs/default.yaml --out reports/contrast_default   # §3.1
python scripts/adapt.py --config configs/default.yaml --out reports/adapt_default       # §3.2
python scripts/eval.py  --config configs/smoke.yaml --out reports/baselines_smoke       # §3.3
python scripts/make_explanations.py --config configs/smoke.yaml --out reports/explain_smoke   # §3.4
python scripts/audit.py --out reports/audit.md                                            # §3 gates
```

Multi-seed (GPU recommended): `python scripts/run_experiment.py --config configs/default.yaml --all --outdir reports/exp_full`