# Final audit (generated — do not hand-edit; rerun `python scripts/audit.py`)

## Checks

| area | status | detail |
|---|---|---|
| leak/norm-guard | PASS | transform() asserts fit() first (train-only) |
| leak/default-unlabeled | PASS | default sup=-1 (labels never touch loss) |
| leak/contrast-no-labels | PASS | ContrastModule steps label-free |
| leak/adapt-no-labels | PASS | AdaptModule step label-free |
| leak/threshold-val-only | PASS | no test-label thresholding |
| repro/synthetic | PASS | same seed -> identical series+labels+masks |
| repro/init | PASS | same seed -> identical init forward |
| gate/pytest | PASS | 61 passed, 2 warnings in 21.71s |
| gate/ruff | PASS | All checks passed! |
| gate/black | PASS | 102 files would be left unchanged. |
| gate/mypy | PASS | Success: no issues found in 63 source files |
| obj/O1 | PASS | code+config+test+exp+figure present (E1) |
| obj/O2 | PASS | code+config+test+exp+figure present (E3) |
| obj/O3 | PASS | code+config+test+exp+figure present (E4) |
| obj/O4 | PASS | code+config+test+exp+figure present (E6) |

## Label use (where labels may appear)

- representation/adaptation losses: label-free (InfoNCE/TS2Vec/masked/MMD/CORAL/DANN).
- prototype bank: normal-train-window SELECTION uses train labels (mirrors 'train assumed normal');
-   flagged fallback logged when no clean windows.
- weak supervision: labels enter ONLY if loss.w_sup>0 AND data.weak_labels>0 (both default 0).
- metrics/thresholds: val labels allowed (best-f1); test labels for scoring only, never fitting.

## Gates

- pytest: PASS — 61 passed, 2 warnings in 21.71s
- ruff: PASS — All checks passed!
- black: PASS — 102 files would be left unchanged.
- mypy: PASS — Success: no issues found in 63 source files

## O1–O4 mapping

| obj | code | config | test | exp | figure | ok |
|---|---|---|---|---|---|---|
| O1 | models/encoders.py, losses/contrastive.py, lightning/contrast.py | configs/model/*.yaml | tests/test_contrast.py | E1 | reports/embed_tcn_domain_0.png | YES |
| O2 | losses/{mmd,coral,dann}.py, alignment/adaptive.py | configs/align/*.yaml | tests/test_alignment.py | E3 | reports/adapt_mmd.png | YES |
| O3 | alignment/{gate,domain_weights,preserve,monitor}.py | configs/align/adaptive_full.yaml | tests/test_alignment.py | E4 | reports/exp/e4_smoke.csv | YES |
| O4 | explain/*.py (6 parts) | configs/explain.yaml | tests/test_explain.py | E6 | reports/explain_smoke/index.html | YES |

## Explicit gaps / not-validated

| obj | item | status |
|---|---|---|
| O1 | weak-label supervision | IMPLEMENTED this phase (SupCon, off by default); single-seed smoke only |
| O2 | conditional/class-prototype alignment | chamfer option implemented, not validated E2E |
| O2 | DANN/CORAL E2E | unit-tested; E3 full-mode only, not run on GPU box yet |
| O3 | multi-seed neg-transfer stats | 2 seeds run; 5-seed full pending (Phase 9 plan) |
| O4 | DeepSHAP | NOT shipped: crashes on BatchNorm TCNs (shap internals); KernelSHAP+LIME+IG cover SHAP/LIME ask |
| O4 | VUS/affiliation metrics | NOT implemented (documented; range/event-wise done) |
| data | real SMD/SMAP transfer numbers | NOT RUN (no local files; loaders ready, graceful skip) |
| data | LODO/multi-source E2E | registry tested; E2 runs pairs only |
| train | MLflow/W&B logging E2E | NOT RUN (mlflow not installed; Tracker unit-covered by smoke off-mode) |
| deploy | `make` on Windows | make binary absent here; python equivalents verified (README) |
