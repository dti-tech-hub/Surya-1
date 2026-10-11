# VIVA_QA.md — 40 likely questions, answered from THIS implementation

## Problem & objectives
1. **What are the three challenges?** Domain shift (detectors fail across machines/patients/segments),
   negative transfer (naive alignment erases anomaly signals), black-box scores (no WHY).
2. **Map O1–O4 to code.** O1: `models/encoders.py` + `losses/contrastive.py` + `lightning/contrast.py` (E1).
   O2: `losses/{mmd,coral,dann}.py` + `alignment/adaptive.py` (E3). O3: gate/weights/preserve/monitor (E4).
   O4: `explain/*` six parts (E6).
3. **What is the core novelty?** Adaptive gating + scale-free domain weights + anomaly-preserving
   constraints + neg-transfer monitor — each ablatable via `configs/align/*.yaml`.

## Data
4. **How is leakage prevented?** Series-level time split → window after split (`splits.py`,
   `assert_window_disjointness` test); norm fit-train-only; val-only thresholds/prototypes.
5. **What is a domain here?** SMD: machine; SMAP/MSL: entity; UCR: series group; synthetic: shift config.
6. **Unlabeled-target protocol?** Adaptation sees target-train windows only, labels never used
   (`DomainRegistry.source_target`, `PairedDataset` carries no labels).
7. **Ground-truth masks?** Only synthetic (`test_channel_mask`) — used to validate explainability (E6).
8. **Why per-split anomaly injection?** Guarantees test positives on tiny series (found zero-anomaly test
   split during Phase 3 debugging).

## Eval
9. **Why report PA if it inflates?** Benchmark comparability; always beside strict with a warning
   (`point_adjust.py`, `report.py` header). Our smoke showed pa 1.0 vs strict 0.12 — the warning proved itself.
10. **Threshold without test labels?** Percentile/POT on normal val; best-F1 on val labels; never test.
11. **What is negative transfer here?** `transfer_report`: gain = adapted − source-only; flag if < 0.
12. **Range/event metrics?** Simplified event-wise (overlap≥1) + Tatbul-style (α-blend); VUS deferred (limitation).

## O1
13. **Why contrastive for anomaly detection?** Learns normal manifold from unlabeled data; anomalies =
    far-from-prototype. Evidence: 3× F1 over random encoder, AUC 0.92, event-recall 1.0.
14. **InfoNCE intuition?** Pull two augmented views together, push other windows apart (temperature τ).
15. **Why max+mean pooling?** Proved mean-only dilution: random-encoder AUC 0.23 (inverted); max preserves spikes.
16. **Prototype vs kNN vs Mahalanobis?** Config `scoring.method`; k-means default (compact), kNN (local),
    Mahalanobis (covariance-aware).

## O2+O3
17. **MMD vs CORAL vs DANN?** MMD = distribution moments via kernels; CORAL = covariance+mean;
    DANN = adversarial domain classifier with GRL. All switchable; MMD default (stable on CPU).
18. **Why median-heuristic bandwidth?** Fixed σ saturated in 64-dim input space (weights stuck at uniform);
    median σ restored 4× related-vs-noise separation.
19. **What does the gate do?** Learns which embedding dims to align (sigmoid + L1/entropy); others stay
    domain-specific. E5 ablates it off.
20. **Domain weights formula?** softmax(−d/mean(d)/τ), τ=1; noise source got 0.22 vs related 0.78.
21. **Anomaly-preserving losses?** Compactness (tight normal) + pseudo-anomaly margin + variance floor
    (no-collapse). Alignment touches normal reps only.
22. **Monitor mechanism?** Val proxies (contrast + compactness); 2 bad epochs → halve α + rollback best weights.
23. **Why did shift-harmful fail?** Per-domain normalization erased mean/var shift → noise domain instead.
24. **Key transfer result?** Seed-0 default-scale: uniform-mixed 0.318→0.000, adaptive 0.444 (best).

## O4
25. **Six explanation parts?** Prototype+medoid overlay, attention heatmap, IG attribution + top channels,
    counterfactual, cross-domain, NL summary — one JSON in `reports/explain_smoke/anomaly_*.json`.
26. **How are attributions validated?** Deletion-AUC (0.5–0.8, IG best), P@k/AUPRC vs GT masks (1.0 on clean hits).
27. **Counterfactual validity?** 10/10 valid; sparsity varies honestly (0.0 whole-window for collective).
28. **Attention with TCN?** Flagged unavailable (no fake weights) — use transformer encoder for attention.
29. **Differentiable score trick?** Softmin prototype distance (`SoftProtoScore`); baseline = medoid window.
30. **Clamp dead-gradient bug?** x on plausibility bound → zero grads; fixed with random delta init.

## Engineering
31. **Reproduce everything?** `make smoke/train/experiments/explain/app/report` (README table).
32. **Determinism?** `set_seed` (random/numpy/torch/cuDNN) + seed in every config; 5-seed aggregation.
33. **Why is `multi_source` tested?** Lint found an undefined-name bug there — untested code hides bugs.
34. **Coverage?** 74%; gaps = file-needing loaders, uninstalled captum, ARIMA fallback (documented).
35. **mypy/ruff/black?** All clean (62 files mypy; ruff with documented BLE/S110/B008 ignores).
36. **CI?** `.github/workflows/ci.yml`: lint + typecheck + pytest + bundle + API tests on CPU.
37. **MLflow/W&B?** `Tracker` (mlflow default, wandb behind flag + `WANDB_API_KEY`, never committed).
38. **API adapt = retraining?** No — prototype-refresh (norm + k-means + p95) on unlabeled CSV; honest scope.
39. **Biggest failure found?** Contrastive F1 0.07 vs iforest 0.61 single-domain (point spikes); strength is
    transfer + recall, stated openly — no SOTA fabrication.
40. **What remains (Phases 9–11)?** Colab runner, report generator, final audit; paper claims need 5-seed
    full runs + real SMD/SMAP numbers.
