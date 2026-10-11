# Limitations (must accompany any submission)

1. Window-pooled embeddings dilute isolated point spikes (max+mean concat mitigates, not solves).
2. Unsupervised-percentile operating point is conservative relative to ranking quality (AUC 0.9+).
3. Negative-transfer effects are seed-dependent on tiny synthetic data; paper claims require 5-seed
   full runs AND real SMD/SMAP numbers (currently synthetic-only for transfer).
4. Point-adjusted metrics are reported beside strict only; PA inflates and must never stand alone.
5. Range/event-wise F1 are simplified (Tatbul-style / overlap>=1); VUS/affiliation not implemented.
6. /adapt is prototype-refresh, not full retraining. Conditional alignment is a chamfer option only.
