"""Time-ordered splits at SERIES level (split first, window later => no window-overlap leakage)."""
from __future__ import annotations
import numpy as np


def time_ordered_bounds(T: int, train_ratio: float = 0.6, val_ratio: float = 0.2) -> tuple[int, int]:
    """Return (train_end, val_end) cut points. Test = [val_end, T). No shuffling."""
    assert 0 < train_ratio < 1 and 0 <= val_ratio < 1 and train_ratio + val_ratio < 1
    tr = int(T * train_ratio)
    va = int(T * (train_ratio + val_ratio))
    return tr, va


def split_series(
    series: np.ndarray, labels: np.ndarray, train_ratio: float = 0.6, val_ratio: float = 0.2
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Split (T, C) series + (T,) labels into (train, val, test) preserving time order."""
    tr, va = time_ordered_bounds(len(series), train_ratio, val_ratio)
    return (
        (series[:tr], labels[:tr]),
        (series[tr:va], labels[tr:va]),
        (series[va:], labels[va:]),
    )


def assert_window_disjointness(T: int, tr: int, va: int, window: int, stride: int) -> None:
    """Verify no window start index is shared across splits (guaranteed by series-level split)."""
    def starts(lo: int, hi: int) -> set[int]:
        return {s for s in range(lo, hi - window + 1, stride)} if hi - lo >= window else set()

    s_tr, s_va, s_te = starts(0, tr), starts(tr, va), starts(va, T)
    assert s_tr.isdisjoint(s_va) and s_tr.isdisjoint(s_te) and s_va.isdisjoint(s_te), "window leakage!"
