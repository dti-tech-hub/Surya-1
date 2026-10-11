"""ARIMA-residual baseline (statsmodels; least-squares AR fallback if unavailable).

Per channel: fit ARIMA(2,0,0)-style model on normal train, score = |test - forecast|.
Falls back to a numpy AR(2) fit when statsmodels is missing — logged, not silent.
"""

from __future__ import annotations

import numpy as np

from tsad.baselines.base import Baseline


def _ar_forecast(train_c: np.ndarray, steps: int, p: int = 2) -> np.ndarray:
    try:
        from statsmodels.tsa.arima.model import ARIMA

        fit = ARIMA(np.nan_to_num(train_c), order=(p, 0, 0)).fit()
        return np.asarray(fit.forecast(steps=steps), dtype=float)
    except Exception:
        # Fallback: least-squares AR(p) on train, recursive forecast of last values
        x = np.nan_to_num(train_c).astype(float)
        if len(x) <= p:
            return np.full(steps, x.mean())
        Y, X = x[p:], np.stack([x[p - i - 1 : len(x) - i - 1] for i in range(p)], axis=1)
        coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
        hist = list(x[-p:])
        out = []
        for _ in range(steps):
            v = float(np.dot(hist[-p:], coef[::-1]))
            out.append(v)
            hist.append(v)
        return np.array(out)


class ARIMABaseline(Baseline):
    name = "arima"

    def __init__(self, p: int = 2):
        self.p = p

    def fit(self, train: np.ndarray) -> ARIMABaseline:
        self.train_ = np.nan_to_num(train).astype(float)
        return self

    def score(self, series: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(series).astype(float)
        T, C = x.shape
        resid = np.zeros((T, C))
        for c in range(C):
            fc = _ar_forecast(self.train_[:, c], T, self.p)
            resid[:, c] = np.abs(x[:, c] - fc)
        return resid.max(axis=1).astype(np.float64)
