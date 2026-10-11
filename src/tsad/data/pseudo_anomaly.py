"""Pseudo-anomaly injection for optional anomaly-aware self-supervision (off by default)."""

from __future__ import annotations

import numpy as np


def inject_pseudo(
    x: np.ndarray, kind: str = "spike", magnitude: float = 3.0, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """x: (T, C) normal window -> (x_aug (T, C), mask (T, C))."""
    rng = np.random.default_rng(seed)
    y, m = x.astype(np.float32).copy(), np.zeros_like(x, dtype=np.int64)
    T, C = y.shape
    c = int(rng.integers(0, C))
    s = int(rng.integers(0, max(1, T - 8)))
    e = min(T, s + int(rng.integers(2, 8)))
    if kind == "spike":
        y[s, c] += magnitude
    elif kind == "scale":
        y[s:e, c] *= 1.0 + magnitude / 5.0
    else:
        raise ValueError(f"unknown pseudo kind {kind}")
    m[s:e, c] = 1
    return y, m
