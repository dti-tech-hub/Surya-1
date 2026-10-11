"""Contrastive + auxiliary losses (config-weighted)."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def info_nce(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """SimCLR NT-Xent over 2B views. z*: (B, p) L2-normalized inside."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)  # (2B, p)
    sim = (z @ z.T) / max(1e-6, tau)
    B = z1.shape[0]
    pos = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    sim.fill_diagonal_(-1e9)  # exclude self
    return F.cross_entropy(sim, pos)


def _instance_temporal(f1: torch.Tensor, f2: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """One scale: instance contrast over batch + temporal contrast over time.

    f*: (B, T, d). Instance: mean-pooled (B, d) cross-view. Temporal: per-series (T, d) cross-view.
    """
    # instance
    li = info_nce(f1.mean(1), f2.mean(1), tau)
    # temporal: for each series, T timesteps contrasted across views
    B, T, _ = f1.shape
    lt = 0.0
    for b in range(B):
        lt = lt + info_nce(f1[b], f2[b], tau)
    return li + lt / max(1, B)


def ts2vec_loss(f1: torch.Tensor, f2: torch.Tensor, tau: float = 0.1,
                max_scales: int = 2) -> torch.Tensor:
    """Hierarchical (TS2Vec-style, simplified): contrast at progressively pooled time scales."""
    loss, w, tot = 0.0, 1.0, 0.0
    a, b = f1, f2
    for _ in range(max_scales):
        loss = loss + w * _instance_temporal(a, b, tau)
        tot += w
        w *= 0.5
        if a.shape[1] < 2:
            break
        a = F.avg_pool1d(a.transpose(1, 2), 2).transpose(1, 2)
        b = F.avg_pool1d(b.transpose(1, 2), 2).transpose(1, 2)
    return loss / max(1e-9, tot)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """pred/target: (B, T, C); mask (B, T) 1 = predict this step."""
    m = mask.unsqueeze(-1).float()
    return (((pred - target) ** 2 * m).sum()) / m.sum().clamp_min(1e-9) / pred.shape[-1]
