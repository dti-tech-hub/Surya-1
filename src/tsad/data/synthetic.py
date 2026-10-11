"""Synthetic multi-domain generator with controllable shift + GT labels/channel masks.

Domain shift knobs per domain: mean, variance scale, frequency, phase, amplitude.
Anomaly types: point spike, contextual, collective/shape, frequency change, level shift, trend.
"""
from __future__ import annotations
import numpy as np
from tsad.data.base import BaseTSADDataset, DomainData
from tsad.data.splits import split_series


ANOMALY_TYPES = ("point", "contextual", "collective", "frequency", "level_shift", "trend")


def _base_series(rng: np.random.Generator, T: int, C: int, freq: float, phase: float, amp: float) -> np.ndarray:
    t = np.arange(T)[:, None]
    ch = np.arange(C)[None, :]
    x = amp * np.sin(2 * np.pi * freq * t / 64.0 + phase + 0.3 * ch)
    x = x + 0.3 * np.sin(2 * np.pi * freq * 2.7 * t / 64.0 + 0.5 * ch)
    return x


def inject_anomalies(
    x: np.ndarray, rng: np.random.Generator, anomaly_ratio: float = 0.05
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inject anomalies into copy of (T, C) series. Returns (x_noisy, labels (T,), mask (T, C))."""
    T, C = x.shape
    labels = np.zeros(T, dtype=np.int64)
    mask = np.zeros((T, C), dtype=np.int64)
    # Budget anomalous POINTS to ~T*ratio (tiny series get fewer event types so splits keep clean windows)
    n_types = 6 if T >= 600 else (3 if T >= 200 else 2)
    kinds = ANOMALY_TYPES[:n_types]
    per_type = max(1, round(T * anomaly_ratio / n_types / 10))
    max_len = max(2, min(30, T // 20))  # event lengths scale with series length (smoke-safe)
    for i, kind in enumerate(kinds):
        for _ in range(per_type):
            c = int(rng.integers(0, C))
            s = int(rng.integers(0, max(1, T - max_len)))
            L = int(rng.integers(1, max_len)) if kind != "point" else 1
            e = min(T, s + L)
            if kind == "point":
                x[s, c] += float(rng.uniform(4, 6)) * (1 if rng.random() > 0.5 else -1)
            elif kind == "contextual":
                x[s:e, c] += float(rng.uniform(1.5, 2.5))
            elif kind == "collective":
                x[s:e, c] += 2.0 * np.sin(np.linspace(0, 3 * np.pi, e - s))
            elif kind == "frequency":
                x[s:e, c] += np.sin(np.linspace(0, 12 * np.pi, e - s))
            elif kind == "level_shift":
                x[s:e, c] += float(rng.uniform(1.0, 2.0))  # bounded shift (length L, smoke-safe)
            elif kind == "trend":
                x[s:e, c] += np.linspace(0, 3.0, e - s)
            labels[s:e] = 1
            mask[s:e, c] = 1
    return x, labels, mask


def generate_domain(
    seed: int,
    T: int = 2000,
    C: int = 4,
    mean: float = 0.0,
    var_scale: float = 1.0,
    freq: float = 1.0,
    phase: float = 0.0,
    amp: float = 1.0,
    anomaly_ratio: float = 0.05,
    waveform: str = "sine",  # sine | noise (noise = structureless harmful domain for E4/Phase-4)
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray],
           tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    if waveform == "noise":
        x = rng.standard_normal((T, C))  # no shared temporal structure
    else:
        x = _base_series(rng, T, C, freq, phase, amp)
        x = x * var_scale + mean + 0.1 * rng.standard_normal((T, C))
    x = x.astype(np.float32)
    # Split FIRST (clean), then inject per split so every split carries ~anomaly_ratio
    # anomalies (guarantees non-empty test positives even on tiny smoke series).
    (tr_x, _), (va_x, _), (te_x, _) = split_series(x, np.zeros(len(x), np.int64))
    tr_x, tr_y, _ = inject_anomalies(tr_x.copy(), np.random.default_rng(seed + 1), anomaly_ratio)
    va_x, va_y, _ = inject_anomalies(va_x.copy(), np.random.default_rng(seed + 2), anomaly_ratio)
    te_x, te_y, te_m = inject_anomalies(te_x.copy(), np.random.default_rng(seed + 3), anomaly_ratio)
    return (tr_x, tr_y), (va_x, va_y), (te_x, te_y, te_m)


class SyntheticDataset(BaseTSADDataset):
    name = "synthetic"

    def __init__(self, n_domains: int = 3, length: int = 2000, n_channels: int = 4,
                 anomaly_ratio: float = 0.05, seed: int = 0, harmful_last: bool = False,
                 harmful_kind: str = "shift"):  # shift | noise (noise = structureless, truly harmful)
        self.n_domains = n_domains
        self.length = length
        self.n_channels = n_channels
        self.anomaly_ratio = anomaly_ratio
        self.seed = seed
        self.harmful_last = harmful_last  # last domain = unrelated distribution (for E4 neg-transfer)
        self.harmful_kind = harmful_kind

    def load(self) -> dict[str, DomainData]:
        out: dict[str, DomainData] = {}
        for k in range(self.n_domains):
            if self.harmful_last and k == self.n_domains - 1:
                if self.harmful_kind == "noise":
                    kw: dict = dict(waveform="noise")  # no shared structure post-normalization
                else:
                    kw = dict(mean=8.0, var_scale=3.0, freq=4.0, phase=2.5, amp=3.0)
            else:
                kw = dict(mean=0.5 * k, var_scale=1.0 + 0.3 * k, freq=1.0 + 0.25 * k,
                          phase=0.4 * k, amp=1.0 + 0.2 * k)
            (tr_x, tr_y), (va_x, va_y), (te_x, te_y, te_m) = generate_domain(
                self.seed + 100 * k, self.length, self.n_channels,
                anomaly_ratio=self.anomaly_ratio, **kw)
            out[f"domain_{k}"] = DomainData(
                domain_id=f"domain_{k}", train_series=tr_x, val_series=va_x, test_series=te_x,
                train_labels=tr_y, val_labels=va_y, test_labels=te_y,
                test_channel_mask=te_m, meta={"shift": kw})
        return out
