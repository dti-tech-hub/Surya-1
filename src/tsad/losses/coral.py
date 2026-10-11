"""CORAL: Frobenius distance between covariances + mean shift. X, Y: (n, d), (m, d)."""
from __future__ import annotations
import torch


def coral(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    def _cov(Z: torch.Tensor) -> torch.Tensor:
        Zc = Z - Z.mean(0, keepdim=True)
        n = max(1, Z.shape[0] - 1)
        return (Zc.T @ Zc) / n

    return torch.norm(_cov(X) - _cov(Y), p="fro") / (X.shape[1] ** 2) \
        + torch.norm(X.mean(0) - Y.mean(0)) ** 2
