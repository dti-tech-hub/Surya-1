"""Common interface for all TSAD datasets. Tensor convention: (T, C) series, (B, T, C) windows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DomainData:
    """One domain's time-ordered data (already split at SERIES level to avoid leakage)."""

    domain_id: str
    train_series: np.ndarray  # (Ttr, C)
    val_series: np.ndarray  # (Tva, C)
    test_series: np.ndarray  # (Tte, C)
    train_labels: np.ndarray  # (Ttr,) 0/1 (train ideally all-normal; kept for checks)
    val_labels: np.ndarray  # (Tva,)
    test_labels: np.ndarray  # (Tte,)
    test_channel_mask: np.ndarray | None = None  # (Tte, C) GT anomalous channels (synthetic only)
    meta: dict = field(default_factory=dict)


class DatasetMissingError(FileNotFoundError):
    """Raised when access-required dataset files are absent. Message must carry download instructions."""


class BaseTSADDataset(ABC):
    """All loaders return dict[domain_id, DomainData]. Normalization/splits live in caller utils."""

    name: str = "base"
    access_required: bool = False
    instructions: str = ""

    @abstractmethod
    def load(self) -> dict[str, DomainData]:
        raise NotImplementedError

    def available(self) -> bool:
        try:
            self.load()
            return True
        except DatasetMissingError:
            return False
