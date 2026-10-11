"""DANN pieces: gradient reversal layer + domain discriminator + lambda schedule."""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x

    @staticmethod
    def backward(ctx, grad_out):
        return -ctx.lambd * grad_out, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Identity forward; backward multiplies grad by -lambd (adversarial)."""
    return _GRL.apply(x, lambd)


def dann_lambda(progress: float) -> float:
    """Ganin schedule: 0 -> 1 over training progress in [0, 1]."""
    p = min(1.0, max(0.0, progress))
    return 2.0 / (1.0 + math.exp(-10 * p)) - 1.0


class DomainDiscriminator(nn.Module):
    def __init__(self, d: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # (N, d) -> (N,) logits
        return self.net(z).squeeze(-1)
