"""Per-domain normalization fit on TRAIN ONLY. Handles NaN + constant channels."""
from __future__ import annotations
import numpy as np


class PerDomainNormalizer:
    """Z-score per channel. fit() sees train only; transform() applies everywhere."""

    def __init__(self, eps: float = 1e-8):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.eps = eps
        self.n_channels_: int = 0

    def fit(self, train: np.ndarray) -> PerDomainNormalizer:
        """train: (T, C). NaN-aware (nanmean/nanstd); constant channels -> std=1."""
        assert train.ndim == 2
        x = np.where(np.isnan(train), np.nan, train)
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        mean = np.where(np.isnan(mean), 0.0, mean)  # all-NaN channel -> 0
        std = np.where(np.isnan(std) | (std < self.eps), 1.0, std)  # constant -> 1
        self.mean_, self.std_ = mean.astype(np.float64), std.astype(np.float64)
        self.n_channels_ = train.shape[1]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "call fit() on train first — no leakage allowed"
        assert x.shape[1] == self.n_channels_, f"channel mismatch {x.shape[1]} vs {self.n_channels_}"
        xn = (np.nan_to_num(x, nan=0.0) - self.mean_) / self.std_
        return np.nan_to_num(xn, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)

    def fit_transform(self, train: np.ndarray) -> np.ndarray:
        return self.fit(train).transform(train)
