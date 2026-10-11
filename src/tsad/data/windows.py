"""Sliding windows over (T, C) series."""
from __future__ import annotations
import numpy as np


def sliding_windows(x: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Slice (T, C) -> (N, W, C). N = floor((T - W) / stride) + 1. Raises if T < W."""
    assert x.ndim == 2, f"expected (T, C), got {x.shape}"
    T, C = x.shape
    if T < window:
        raise ValueError(f"series length {T} < window {window}")
    starts = range(0, T - window + 1, stride)
    return np.stack([x[s : s + window] for s in starts], axis=0)


def window_label(labels: np.ndarray, window: int, stride: int, mode: str = "any") -> np.ndarray:
    """Aggregate point labels (T,) -> window labels (N,). mode: any|max (same) or mean ratio."""
    w = sliding_windows(labels[:, None], window, stride)[:, :, 0]
    if mode in ("any", "max"):
        return (w.max(axis=1) > 0).astype(np.int64)
    if mode == "mean":
        return w.mean(axis=1)
    raise ValueError(f"unknown mode {mode}")


def window_mask(mask: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Aggregate channel mask (T, C) -> (N, W, C) windows."""
    assert mask.ndim == 2
    T, C = mask.shape
    starts = list(range(0, T - window + 1, stride))
    return np.stack([mask[s : s + window] for s in starts], axis=0)
