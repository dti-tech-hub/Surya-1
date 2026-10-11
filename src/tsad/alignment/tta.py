"""Test-time adaptation baseline: AdaBN — refresh BatchNorm stats on unlabeled target data.

No labels, no gradients: run the encoder over target-train windows in train mode, then eval.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


@torch.no_grad()
def adabn_adapt(module: nn.Module, X: np.ndarray, batch: int = 256) -> nn.Module:
    """Update BN running stats from target windows X (N, T, C). Returns module (eval mode)."""
    was_training = module.training
    module.train()  # enable BN stat updates (dropout absent in encoders; no grad)
    dev = next(module.parameters()).device
    enc = getattr(module, "enc", module)  # Lightning module wraps .enc; plain encoder is itself
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i : i + batch].astype(np.float32)).to(dev)
        _ = enc(xb)
    module.train(was_training)
    return module.eval()
