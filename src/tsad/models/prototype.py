"""Prototype bank over NORMAL train embeddings + kNN / Mahalanobis options."""
from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans


class PrototypeBank:
    """fit(Z_normal (N, d)); score(Z (M, d)) -> dist to nearest prototype (M,)."""

    def __init__(self, k: int = 8, method: str = "kmeans", seed: int = 0):
        assert method in ("kmeans", "knn", "mahalanobis")
        self.k, self.method, self.seed = k, method, seed

    def fit(self, Z: np.ndarray) -> PrototypeBank:
        Z = np.asarray(Z, dtype=np.float64)
        assert len(Z) >= 1
        if self.method == "kmeans":
            k = min(self.k, len(Z))
            self.proto_ = KMeans(n_clusters=k, n_init=10, random_state=self.seed).fit(Z).cluster_centers_
        elif self.method == "knn":
            self.proto_ = Z  # keep all; score = mean dist to k nearest
        else:
            self.mean_ = Z.mean(0)
            cov = np.cov(Z.T) + 1e-6 * np.eye(Z.shape[1])
            self.inv_ = np.linalg.inv(cov)
        return self

    def score(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=np.float64)
        if self.method == "mahalanobis":
            d = Z - self.mean_
            return np.sqrt(np.einsum("ij,jk,ik->i", d, self.inv_, d))
        D = np.linalg.norm(Z[:, None, :] - self.proto_[None, :, :], axis=2)
        if self.method == "kmeans":
            return D.min(axis=1)
        k = min(self.k, D.shape[1])  # knn: mean of k nearest
        return np.sort(D, axis=1)[:, :k].mean(axis=1)

    @property
    def prototypes(self) -> np.ndarray | None:
        return getattr(self, "proto_", None)
