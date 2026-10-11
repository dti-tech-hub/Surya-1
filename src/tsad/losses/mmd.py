"""Multi-kernel MMD (biased V-statistic). X, Y: (n, d), (m, d)."""
from __future__ import annotations
import torch


def rbf_kernel(X: torch.Tensor, Y: torch.Tensor, sigma: float) -> torch.Tensor:
    XX = (X * X).sum(1, keepdim=True)
    YY = (Y * Y).sum(1, keepdim=True).T
    D2 = (XX + YY - 2 * X @ Y.T).clamp_min(0.0)
    return torch.exp(-D2 / (2 * sigma * sigma + 1e-12))


def _median_bandwidth(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Median heuristic: median pairwise distance on a subsample (scale-free RBF)."""
    with torch.no_grad():
        n = min(256, len(X), len(Y))
        A, B = X[:n], Y[:n]
        D2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        med = float(torch.median(D2).sqrt())
        return max(1e-3, med)


def mmd2(X: torch.Tensor, Y: torch.Tensor,
         sigmas: tuple[float, ...] | None = None) -> torch.Tensor:
    """Squared multi-kernel MMD. sigmas=None -> median heuristic (0.5x, 1x, 2x)."""
    if sigmas is None:
        m = _median_bandwidth(X, Y)
        sigmas = (0.5 * m, m, 2.0 * m)
    Kxx = sum(rbf_kernel(X, X, s) for s in sigmas) / len(sigmas)
    Kyy = sum(rbf_kernel(Y, Y, s) for s in sigmas) / len(sigmas)
    Kxy = sum(rbf_kernel(X, Y, s) for s in sigmas) / len(sigmas)
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
