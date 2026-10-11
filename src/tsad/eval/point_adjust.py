"""Point-adjusted (PA) protocol. WARNING: PA inflates scores — always report strict alongside."""
from __future__ import annotations
import numpy as np


def point_adjust(scores: np.ndarray, labels: np.ndarray, threshold: float) -> np.ndarray:
    """PA: if ANY point in a GT anomalous event exceeds threshold, mark the WHOLE event detected.

    scores: (T,), labels: (T,) binary. Returns adjusted binary preds (T,).
    """
    pred = (scores >= threshold).astype(np.int64)
    adj = pred.copy()
    T = len(labels)
    i = 0
    while i < T:
        if labels[i] == 1:
            j = i
            while j < T and labels[j] == 1:
                j += 1
            if pred[i:j].max() > 0:  # one hit -> whole event counts
                adj[i:j] = 1
            i = j
        else:
            i += 1
    return adj
