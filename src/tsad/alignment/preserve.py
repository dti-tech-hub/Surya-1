"""C. Anomaly-preserving constraints (align NORMAL only + keep separable structure).

- compactness: normal embeddings stay tight (mean dist to batch centroid).
- margin: pseudo-anomaly embeddings stay >= margin above normal scores (hinge).
- no_collapse: total variance must not fall below a floor (alignment must not erase
  variance directions that carry anomaly information).
"""

from __future__ import annotations

import torch


def compactness(Z: torch.Tensor) -> torch.Tensor:
    """Mean squared distance to batch centroid. Z: (N, d) normal-only reps."""
    c = Z.mean(0, keepdim=True).detach()  # anchor fixed per step (no trivial shrink)
    return ((Z - c) ** 2).mean()


def pseudo_margin(
    normal_scores: torch.Tensor, pseudo_scores: torch.Tensor, margin: float = 1.0
) -> torch.Tensor:
    """Hinge: pseudo-anomaly scores must exceed normal mean by >= margin."""
    return torch.clamp(margin - (pseudo_scores.mean() - normal_scores.mean()), min=0.0)


def no_collapse(Z: torch.Tensor, floor: float = 1e-3) -> torch.Tensor:
    """Penalize total per-dim variance falling below floor (prevents alignment collapse)."""
    v = Z.var(0).mean()
    return torch.clamp(floor - v, min=0.0)


def preserve_loss(
    Z_normal: torch.Tensor,
    pseudo_scores: torch.Tensor | None = None,
    normal_scores: torch.Tensor | None = None,
    beta_c: float = 1.0,
    beta_m: float = 1.0,
    beta_v: float = 1.0,
) -> torch.Tensor:
    out = beta_c * compactness(Z_normal) + beta_v * no_collapse(Z_normal)
    if pseudo_scores is not None and normal_scores is not None:
        out = out + beta_m * pseudo_margin(normal_scores, pseudo_scores)
    return out
