"""Threshold selection — fitted WITHOUT test labels (val only).

- best_f1: needs val LABELS (supervised val, standard for SMD-style benchmarks).
- percentile / pot: unsupervised, fit on NORMAL val scores only.
- fixed: literal value from config.
"""
from __future__ import annotations
import numpy as np
from tsad.eval.metrics import prf_from_pred


def best_f1_threshold(val_labels: np.ndarray, val_scores: np.ndarray,
                      grid: int = 100) -> float:
    val_scores = np.asarray(val_scores, dtype=float)
    cands = np.unique(np.quantile(val_scores, np.linspace(0, 1, grid)))
    best_t, best_f = float(cands[-1]), -1.0
    for t in cands:
        f = prf_from_pred(np.asarray(val_labels), (val_scores >= t).astype(int))["f1"]
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t


def percentile_threshold(normal_scores: np.ndarray, percentile: float = 95.0) -> float:
    return float(np.quantile(np.asarray(normal_scores, dtype=float), percentile / 100.0))


def pot_threshold(normal_scores: np.ndarray, q: float = 0.98, risk: float = 1e-4) -> float:
    """Simplified POT (Siffer et al. 2017): GPD fit on exceedances over init quantile.

    Falls back to the init quantile if scipy is missing or fit fails. risk=q-level target.
    """
    s = np.asarray(normal_scores, dtype=float)
    t0 = float(np.quantile(s, q))
    exc = s[s > t0] - t0
    if len(exc) < 10:
        return t0
    try:
        from scipy.stats import genpareto

        c, _, scale = genpareto.fit(exc, floc=0)
        n, n_exc = len(s), len(exc)
        level = t0 + (scale / c) * (((n / n_exc) * risk) ** (-c) - 1) if abs(c) > 1e-6 \
            else t0 - scale * np.log((n / n_exc) * risk)
        return float(level) if np.isfinite(level) else t0
    except Exception:
        return t0


def select_threshold(method: str, val_scores: np.ndarray, val_labels: np.ndarray | None = None,
                     percentile: float = 95.0, fixed: float = 0.5) -> float:
    """Dispatch. val_labels required only for best_f1; others use scores (normal val) only."""
    if method == "best-f1":
        assert val_labels is not None, "best-f1 needs val labels"
        return best_f1_threshold(val_labels, val_scores)
    if method == "percentile":
        return percentile_threshold(val_scores, percentile)
    if method == "pot":
        return pot_threshold(val_scores)
    if method == "fixed":
        return float(fixed)
    raise ValueError(f"unknown threshold method {method}")
