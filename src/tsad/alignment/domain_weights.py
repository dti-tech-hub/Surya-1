"""B. Domain-level adaptive weighting: w_k ~ softmax(-dist_k / tau).

dist_k = MMD (default) / proxy A-distance / embedding-stats distance between source k and target.
Dissimilar sources get ~0 weight (negative-transfer prevention). Learnable temperature tau.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from tsad.losses.mmd import mmd2


def stats_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Cheap embedding-statistics distance: mean shift + std-ratio gap."""
    mx, my = X.mean(0), Y.mean(0)
    sx, sy = X.std(0) + 1e-6, Y.std(0) + 1e-6
    return torch.norm(mx - my) + torch.norm(torch.log(sx / sy))


def proxy_a_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Proxy A-distance via a linear domain classifier error: d = 2(1 - 2*err). Differentiable-ish.

    Solved in closed form (least squares), detached — used as a weighting signal only.
    """
    with torch.no_grad():
        Z = torch.cat([X, Y], dim=0).detach().cpu().numpy()
        lab = np.concatenate([np.zeros(len(X)), np.ones(len(Y))])
        Z1 = np.concatenate([Z, np.ones((len(Z), 1))], axis=1)
        coef, *_ = np.linalg.lstsq(Z1, lab, rcond=None)
        err = float(np.mean(((Z1 @ coef > 0.5).astype(float) - lab) ** 2))
        err = min(0.5, max(0.0, err))
    return torch.tensor(2 * (1 - 2 * err), dtype=torch.float32)


class DomainWeighter(nn.Module):
    def __init__(self, metric: str = "mmd", tau: float = 1.0):
        super().__init__()
        assert metric in ("mmd", "proxy_a", "stats")
        self.metric = metric
        self.log_tau = nn.Parameter(torch.tensor(float(np.log(max(1e-3, tau)))))
        super().__init__()
        assert metric in ("mmd", "proxy_a", "stats")
        self.metric = metric
        self.log_tau = nn.Parameter(torch.tensor(float(np.log(max(1e-3, tau)))))

    def distances(self, srcs: list[torch.Tensor], tgt: torch.Tensor) -> torch.Tensor:
        ds = []
        for s in srcs:
            if self.metric == "mmd":
                ds.append(mmd2(s, tgt).detach())
            elif self.metric == "proxy_a":
                ds.append(proxy_a_distance(s, tgt).to(s.device))
            else:
                ds.append(stats_distance(s, tgt).detach())
        return torch.stack(ds)

    def forward(self, srcs: list[torch.Tensor], tgt: torch.Tensor) -> torch.Tensor:
        """-> w (K,) summing to 1. Single source -> [1]. Scale-free: distances normalized by mean."""
        if len(srcs) == 1:
            return torch.ones(1, device=tgt.device)
        tau = torch.exp(self.log_tau).clamp_min(1e-3)
        d = self.distances(srcs, tgt)
        d = d / d.mean().clamp_min(1e-9).detach()  # scale-free sharpness
        return torch.softmax(-d / tau, dim=0)
