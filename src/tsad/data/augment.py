"""Time-series augmentations. Each fn: (T, C) float32 -> (T, C) float32. Pipeline is config-driven."""
from __future__ import annotations
import numpy as np


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def jitter(x: np.ndarray, sigma: float = 0.05, seed: int | None = None) -> np.ndarray:
    return (x + _rng(seed).normal(0, sigma, x.shape)).astype(np.float32)


def scaling(x: np.ndarray, lo: float = 0.9, hi: float = 1.1, seed: int | None = None) -> np.ndarray:
    r = _rng(seed).uniform(lo, hi, (1, x.shape[1]))
    return (x * r).astype(np.float32)


def magnitude_warp(x: np.ndarray, knots: int = 4, sigma: float = 0.2, seed: int | None = None) -> np.ndarray:
    rng = _rng(seed)
    T, C = x.shape
    warp = np.ones((T, C), dtype=np.float32)
    xs = np.linspace(0, T - 1, knots + 2)
    for c in range(C):
        ys = 1.0 + rng.normal(0, sigma, knots + 2)
        warp[:, c] = np.interp(np.arange(T), xs, ys)
    return (x * warp).astype(np.float32)


def time_warp(x: np.ndarray, knots: int = 4, sigma: float = 0.1, seed: int | None = None) -> np.ndarray:
    rng = _rng(seed)
    T, C = x.shape
    tt = np.arange(T, dtype=float)
    warp = tt + np.interp(tt, np.linspace(0, T - 1, knots + 2), rng.normal(0, sigma * T, knots + 2))
    warp = np.clip(warp, 0, T - 1)
    return np.stack([np.interp(warp, tt, x[:, c]) for c in range(C)], axis=1).astype(np.float32)


def window_crop(x: np.ndarray, crop_ratio: float = 0.9, seed: int | None = None) -> np.ndarray:
    rng = _rng(seed)
    T = len(x)
    L = max(1, int(T * crop_ratio))
    s = int(rng.integers(0, T - L + 1)) if T > L else 0
    seg = x[s:s + L]
    idx = np.linspace(0, L - 1, T).astype(int)  # resize back to T
    return seg[idx].astype(np.float32)


def masking(x: np.ndarray, mask_ratio: float = 0.1, seed: int | None = None) -> np.ndarray:
    rng = _rng(seed)
    m = rng.random(x.shape) < mask_ratio
    y = x.copy()
    y[m] = 0.0
    return y.astype(np.float32)


def permutation(x: np.ndarray, n_segments: int = 4, seed: int | None = None) -> np.ndarray:
    rng = _rng(seed)
    T = len(x)
    edges = sorted(rng.choice(T, n_segments - 1, replace=False)) if T > n_segments else []
    bounds = [0, *map(int, edges), T]
    segs = [x[bounds[i]:bounds[i + 1]] for i in range(n_segments)]
    rng.shuffle(segs)
    return np.concatenate(segs, axis=0).astype(np.float32)


def freq_perturb(x: np.ndarray, sigma: float = 0.05, seed: int | None = None) -> np.ndarray:
    rng = _rng(seed)
    F = np.fft.rfft(x, axis=0)
    F = F + sigma * (rng.standard_normal(F.shape) + 1j * rng.standard_normal(F.shape)) * np.abs(F).mean()
    return np.fft.irfft(F, n=len(x), axis=0).astype(np.float32)


def channel_dropout(x: np.ndarray, p: float = 0.2, seed: int | None = None) -> np.ndarray:
    rng = _rng(seed)
    drop = rng.random(x.shape[1]) < p
    if drop.all():
        drop[int(rng.integers(0, x.shape[1]))] = False  # keep at least one
    y = x.copy()
    y[:, drop] = 0.0
    return y.astype(np.float32)


AUGMENTS = {"jitter": jitter, "scaling": scaling, "magnitude_warp": magnitude_warp,
            "time_warp": time_warp, "window_crop": window_crop, "masking": masking,
            "permutation": permutation, "freq_perturb": freq_perturb, "channel_dropout": channel_dropout}


class AugmentPipeline:
    """Configurable: AugmentPipeline(["jitter", "scaling"]) applies in order."""

    def __init__(self, names: list[str] | None = None, seed: int | None = None):
        self.names = names or []
        for n in self.names:
            if n not in AUGMENTS:
                raise ValueError(f"unknown augment {n}; choose from {sorted(AUGMENTS)}")
        self.seed = seed

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=np.float32)
        for i, n in enumerate(self.names):
            y = AUGMENTS[n](y, seed=None if self.seed is None else self.seed + i)
        return y.astype(np.float32)

    def two_views(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self(x), self(x)
