"""D. Negative-transfer monitor: watch unlabeled val proxies, adapt alpha, rollback if hurt.

Proxies (no labels): source-val contrastive loss + target-val compactness.
Worse-than-baseline for `patience` epochs -> halve alpha (floor 0); keep best snapshot for rollback.
"""

from __future__ import annotations

import copy

from torch import nn


class NegTransferMonitor:
    def __init__(self, patience: int = 2, factor: float = 0.5, min_alpha: float = 0.0):
        self.patience, self.factor, self.min_alpha = patience, factor, min_alpha
        self.best: float | None = None
        self.bad = 0
        self.alpha = 1.0  # multiplier on configured alpha
        self.snapshot: dict | None = None

    def proxy(self, val_contrast: float, tgt_compact: float, w: float = 0.5) -> float:
        return w * val_contrast + (1 - w) * tgt_compact

    def update(self, module: nn.Module, val_contrast: float, tgt_compact: float) -> dict:
        p = self.proxy(val_contrast, tgt_compact)
        info = {"proxy": p, "alpha_mult": self.alpha, "rolled_back": False}
        if self.best is None or p < self.best:
            self.best = p
            self.bad = 0
            self.snapshot = copy.deepcopy({k: v.cpu() for k, v in module.state_dict().items()})
        else:
            self.bad += 1
            if self.bad >= self.patience:
                self.alpha = max(self.min_alpha, self.alpha * self.factor)
                self.bad = 0
                if self.snapshot is not None:  # rollback to best-known weights
                    module.load_state_dict(
                        {k: v.to(module.state_dict()[k].device) for k, v in self.snapshot.items()}
                    )
                    info["rolled_back"] = True
                info["alpha_mult"] = self.alpha
        return info
