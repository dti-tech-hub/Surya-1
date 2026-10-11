# CONFERENCE.md — paper plan (workshop/short paper first)

## Claim (evidence-backed, narrow)
Adaptive feature-gated + source-weighted alignment avoids negative transfer where uniform alignment
collapses, while giving per-flag explanations with measured fidelity. Evidence today: synthetic,
seed-0 default-scale (uniform-mixed 0.318→0.000, adaptive 0.444, related weight 0.67) + smoke replication
+ explainer metrics (IG deletion-AUC ~0.8, CF validity 10/10). NOT claimed: SOTA dominance, real-data
transfer (needs SMD/SMAP runs), 5-seed significance (needs full runs).

## Venue targets
1. Workshop/short paper (applied ML / time-series / AIOps workshop at KDD/ICDM/AAAI): fits honest scope.
2. Full: IEEE BigData / IJCNN / ECAI-PAAD style venue after real-data + 5-seed results.

## Paper skeleton -> repo assets (`make report` -> `reports/final/`)
- Intro/motivation: three challenges (§PROJECT). Related: OmniAnomaly, TS2Vec, DANN, Tatbul metrics.
- Method: O1 contrast → O2/O3 adaptive (Eq. in DESIGN) → O4 six-part explanations.
- Experiments: E1 table (`e1_smoke.tex`), transfer heatmap (E2), E3 bars (`summary_alignment.png`),
  E4 dissimilarity curve, E5 ablations (`ablation_summary.md`), E6 explainer table, E7/E8.
- Results prose: start from `results_section.md` (real numbers only). Limitations: `limitations.md` verbatim.

## TODO before submission
- [ ] 5-seed default runs (GPU box, Phase 9 notebooks) + paired tests.
- [ ] SMD single-domain + SMAP/MSL transfer numbers (needs downloads, Phase 1 notes).
- [ ] Replace every NOT RUN; re-export `reports/final/`.
- [ ] Anonymize repo copy if double-blind (remove git history/authors).
