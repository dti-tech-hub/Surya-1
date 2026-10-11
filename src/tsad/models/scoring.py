"""Window-score -> timestamp-score mapping, smoothing, hybrid combination."""

from __future__ import annotations

import numpy as np


def map_to_timestamps(win_scores: np.ndarray, starts: list[int], T: int, window: int) -> np.ndarray:
    """Mean-aggregate window scores (N,) over covered timesteps -> (T,)."""
    acc, cnt = np.zeros(T), np.zeros(T)
    for s, v in zip(starts, np.asarray(win_scores, dtype=float)):
        e = min(T, s + window)
        acc[s:e] += v
        cnt[s:e] += 1
    return (acc / np.maximum(1, cnt)).astype(np.float64)


def smooth(scores: np.ndarray, width: int = 5) -> np.ndarray:
    """Centered moving average (width>=1); preserves length."""
    scores = np.asarray(scores, dtype=float)
    if width <= 1:
        return scores
    k = np.ones(width) / width
    return np.convolve(scores, k, mode="same")


def zscore_norm(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    mu, sd = s.mean(), s.std()
    return (s - mu) / (sd + 1e-8)


def hybrid(emb_scores: np.ndarray, aux_scores: np.ndarray | None, w: float = 0.7) -> np.ndarray:
    """Normalized hybrid. If aux (e.g. recon error) missing -> pure embedding (warned by caller)."""
    e = zscore_norm(emb_scores)
    if aux_scores is None:
        return e
    return w * e + (1 - w) * zscore_norm(aux_scores)
