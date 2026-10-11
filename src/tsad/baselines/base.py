"""Baseline interface: fit on NORMAL train (T, C) -> score (T,) higher = more anomalous."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Baseline(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, train: np.ndarray) -> Baseline: ...

    @abstractmethod
    def score(self, series: np.ndarray) -> np.ndarray:
        """series: (T, C) -> scores (T,)."""
        ...
