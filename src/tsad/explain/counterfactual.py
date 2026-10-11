"""4. Counterfactual explanation: minimal sparse bounded perturbation pushing score below threshold."""
from __future__ import annotations
import numpy as np
import torch


def counterfactual(score_fn, x: np.ndarray, threshold: float, l1: float = 0.05,
                   steps: int = 100, lr: float = 0.05, lo: float | None = None,
                   hi: float | None = None, seed: int = 0) -> dict:
    """Gradient descent on delta: hinge(score(x+delta) < thr) + l1*|delta|_1, box-clipped.

    lo/hi (plausibility bounds) default to train min/max TXT-passed by caller; fallback = x range.
    Returns dict with cf window, delta, changed channels/steps, validity/sparsity/proximity.
    """
    torch.manual_seed(seed)
    score_fn.eval()
    x = x.astype(np.float32)
    lo = float(x.min()) if lo is None else float(lo)
    hi = float(x.max()) if hi is None else float(hi)
    if hi <= lo:  # degenerate bounds (e.g. constant window) -> widen symmetrically
        lo, hi = lo - 1.0, hi + 1.0
    # Small random init: avoids exact clamp-boundary dead gradients when x sits on lo/hi.
    delta = (torch.zeros_like(torch.tensor(x)) +
             1e-3 * torch.randn_like(torch.tensor(x))).requires_grad_(True)
    opt = torch.optim.Adam([delta], lr=lr)
    xt = torch.tensor(x)
    for _ in range(steps):
        opt.zero_grad()
        xc = (xt + delta).clamp(float(lo), float(hi))
        s = score_fn(xc[None])
        loss = torch.clamp(s - threshold + 0.05, min=0.0).sum() + l1 * delta.abs().mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        cf = (xt + delta).clamp(float(lo), float(hi)).numpy()
    d = (cf - x)
    s_cf = float(score_fn(torch.tensor(cf[None])).item())
    nz = np.abs(d) > 1e-3
    per_ch = np.abs(d).sum(axis=0)
    return {
        "cf_window": cf.astype(float).tolist(),
        "delta_l1": float(np.abs(d).sum()),
        "delta_l2": float(np.sqrt((d ** 2).sum())),
        "frac_changed": float(nz.mean()),
        "score_after": s_cf,
        "valid": bool(s_cf < threshold),
        "top_channels": [int(i) for i in np.argsort(-per_ch)[:3]],
        "per_channel_change": [float(v) for v in per_ch],
        "bounds": [float(lo), float(hi)],
    }
