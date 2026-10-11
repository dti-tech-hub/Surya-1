"""E. Combined adaptive objective.

L = L_contrast + alpha_t * sum_k w_k * L_align(k, gate) + beta * L_preserve + gamma * L_sparse
alpha_t: linear warm-up, then multiplied by the monitor's adaptive factor.
"""

from __future__ import annotations

import torch
from torch import nn

from tsad.alignment.domain_weights import DomainWeighter
from tsad.alignment.gate import FeatureGate
from tsad.alignment.preserve import preserve_loss
from tsad.losses.coral import coral
from tsad.losses.dann import grad_reverse
from tsad.losses.mmd import mmd2


class AdaptiveAligner(nn.Module):
    def __init__(
        self,
        d: int,
        method: str = "mmd",
        use_gate: bool = True,
        gate_kind: str = "sigmoid",
        use_weights: bool = True,
        weight_metric: str = "mmd",
        tau: float = 1.0,
        alpha: float = 0.5,
        beta: float = 0.1,
        gamma: float = 0.01,
        warmup_epochs: int = 1,
        cond: bool = False,
    ):
        super().__init__()
        assert method in ("mmd", "coral", "dann", "none")
        self.method = method
        self.gate = FeatureGate(d, gate_kind) if use_gate else None
        self.weighter = DomainWeighter(weight_metric, tau) if use_weights else None
        self.disc = (
            nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1))
            if method == "dann"
            else None
        )
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.warmup = warmup_epochs
        self.cond = cond
        self.alpha_mult = 1.0  # set by monitor

    def _apply_gate(self, z: torch.Tensor) -> torch.Tensor:
        return self.gate(z) if self.gate is not None else z

    def align_term(self, S: torch.Tensor, T: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
        Sg, Tg = self._apply_gate(S), self._apply_gate(T)
        if self.method == "mmd":
            return mmd2(Sg, Tg)
        if self.method == "coral":
            return coral(Sg, Tg)
        if self.method == "dann":
            from torch.nn.functional import binary_cross_entropy_with_logits

            assert self.disc is not None  # constructed exactly when method == "dann"
            z = torch.cat([Sg, Tg], dim=0)
            y = torch.cat([torch.zeros(len(Sg)), torch.ones(len(Tg))]).to(z.device)
            return binary_cross_entropy_with_logits(self.disc(grad_reverse(z, lambd)), y)
        return torch.zeros((), device=S.device)

    def cond_term(self, S: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Optional conditional alignment: each target rep near some source rep (chamfer)."""
        if not self.cond:
            return torch.zeros((), device=S.device)
        Sg, Tg = self._apply_gate(S), self._apply_gate(T)
        D = torch.cdist(Tg, Sg)
        return D.min(dim=1).values.mean()

    def forward(
        self,
        srcs: list[torch.Tensor],
        tgt: torch.Tensor,
        epoch: int = 0,
        max_epochs: int = 1,
        lambd: float = 1.0,
        pseudo: torch.Tensor | None = None,
        fixed_weights: torch.Tensor | None = None,
    ) -> dict:
        """srcs: per-source NORMAL reps [(n_k, d)]; tgt: target reps (m, d), all unlabeled-target-train."""
        if self.method == "none" or not srcs:
            w = torch.ones(len(srcs)) if srcs else torch.zeros(0)
            z = torch.zeros(())
            return {
                "align": z,
                "weights": w,
                "preserve": z,
                "sparse": z,
                "alpha_t": 0.0,
                "total_align": z,
            }
        if fixed_weights is not None:  # stable epoch-level weights (input-space distances)
            w = fixed_weights.to(srcs[0].device)
        else:
            w = (
                self.weighter(srcs, tgt)
                if self.weighter is not None
                else torch.ones(len(srcs)) / max(1, len(srcs))
            )
        terms = torch.stack([self.align_term(s, tgt, lambd) + self.cond_term(s, tgt) for s in srcs])
        align = (w.to(terms.device) * terms).sum()
        S_all = torch.cat(srcs, dim=0)
        preserve = preserve_loss(self._apply_gate(S_all))
        sparse = self.gate.sparsity_loss() if self.gate is not None else torch.zeros(())
        alpha_t = self.alpha * min(1.0, (epoch + 1) / max(1, self.warmup)) * self.alpha_mult
        total_align = alpha_t * align + self.beta * preserve + self.gamma * sparse
        return {
            "align": align,
            "weights": w.detach(),
            "preserve": preserve.detach(),
            "sparse": sparse.detach() if torch.is_tensor(sparse) else sparse,
            "alpha_t": alpha_t,
            "total_align": total_align,
        }
