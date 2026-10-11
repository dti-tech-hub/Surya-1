"""A. Feature-level adaptive gating: g(z) in [0,1]^d selects aligned dimensions.

Sigmoid gate (default) or hard-concrete (L0-style). Sparsity/entropy reg keeps selection meaningful:
only invariant dims are aligned; domain-specific dims pass through unaligned.
"""

from __future__ import annotations

import torch
from torch import nn


class FeatureGate(nn.Module):
    def __init__(self, d: int, kind: str = "sigmoid", init_bias: float = 0.0):
        super().__init__()
        assert kind in ("sigmoid", "hard_concrete")
        self.kind = kind
        self.logits = nn.Parameter(torch.full((d,), init_bias))

    def forward(self, z: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """z: (N, d) -> gated (N, d). Gate broadcast over batch (global feature selection)."""
        if self.kind == "sigmoid":
            g = torch.sigmoid(self.logits)
        else:  # hard-concrete (Louizos): stretched binary-concrete, eval = deterministic clip
            if self.training and not hard:
                u = torch.rand_like(self.logits).clamp_(1e-6, 1 - 1e-6)
                s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.logits) / 0.5)
                g = s * (1.1 - (-0.1)) + (-0.1)
                g = g.clamp_(0, 1)
            else:
                g = torch.sigmoid(self.logits) * 1.2 - 0.1
                g = g.clamp(0, 1)
        assert ((g >= 0) & (g <= 1)).all()
        return z * g.unsqueeze(0)

    @property
    def probs(self) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.logits).clamp(0, 1)

    def sparsity_loss(self) -> torch.Tensor:
        """L1 on gate probs (prefer few aligned dims) + entropy (prefer decisive 0/1)."""
        p = torch.sigmoid(self.logits).clamp(1e-6, 1 - 1e-6)
        ent = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
        return p.mean() + 0.1 * ent
