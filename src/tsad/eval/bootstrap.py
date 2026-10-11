"""Bootstrap CIs + multi-seed mean±std aggregation."""

from __future__ import annotations

import numpy as np

from tsad.eval.metrics import prf_from_pred


def bootstrap_ci(
    y: np.ndarray,
    pred: np.ndarray,
    metric: str = "f1",
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Resample (T,) pairs with replacement; percentile CI over metric."""
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y).ravel(), np.asarray(pred).ravel()
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        vals.append(prf_from_pred(y[idx], pred[idx])[metric])
    lo, hi = np.quantile(vals, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"mean": float(np.mean(vals)), "lo": float(lo), "hi": float(hi), "n_boot": n_boot}


def aggregate_seeds(rows: list[dict]) -> dict:
    """rows: per-seed metric dicts -> {metric: 'mean±std'} + raw means."""
    keys = sorted({k for r in rows for k in r if isinstance(r[k], (int, float))})
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = float(v.mean())
        out[k + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    out["n_seeds"] = len(rows)
    return out
