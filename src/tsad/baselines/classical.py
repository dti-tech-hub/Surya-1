"""Classical baselines: z-score, IsolationForest, One-Class SVM, PCA reconstruction."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from tsad.baselines.base import Baseline


class ZScoreBaseline(Baseline):
    name = "zscore"

    def fit(self, train: np.ndarray) -> ZScoreBaseline:
        self.mean_ = np.nanmean(train, axis=0)
        std = np.nanstd(train, axis=0)
        self.std_ = np.where(std < 1e-8, 1.0, std)
        return self

    def score(self, series: np.ndarray) -> np.ndarray:
        z = np.abs((np.nan_to_num(series) - self.mean_) / self.std_)
        return z.max(axis=1).astype(np.float64)  # (T,)


class IsolationForestBaseline(Baseline):
    name = "iforest"

    def __init__(self, contamination: float = 0.05, seed: int = 0):
        self.contamination = contamination
        self.seed = seed

    def fit(self, train: np.ndarray) -> IsolationForestBaseline:
        self.clf_ = IsolationForest(contamination=self.contamination, random_state=self.seed)
        self.clf_.fit(np.nan_to_num(train))
        return self

    def score(self, series: np.ndarray) -> np.ndarray:
        return (-self.clf_.decision_function(np.nan_to_num(series))).astype(np.float64)


class OCSVMBaseline(Baseline):
    name = "ocsvm"

    def __init__(self, nu: float = 0.05):
        self.nu = nu

    def fit(self, train: np.ndarray) -> OCSVMBaseline:
        self.clf_ = OneClassSVM(nu=self.nu, kernel="rbf", gamma="scale")
        self.clf_.fit(np.nan_to_num(train))
        return self

    def score(self, series: np.ndarray) -> np.ndarray:
        return (-self.clf_.decision_function(np.nan_to_num(series))).astype(np.float64)


class PCABaseline(Baseline):
    name = "pca"

    def __init__(self, n_components: float = 0.95):
        self.n_components = n_components

    def fit(self, train: np.ndarray) -> PCABaseline:
        self.pca_ = PCA(n_components=self.n_components)
        self.pca_.fit(np.nan_to_num(train))
        return self

    def score(self, series: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(series)
        rec = self.pca_.inverse_transform(self.pca_.transform(x))
        return np.linalg.norm(x - rec, axis=1).astype(np.float64)
