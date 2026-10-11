"""3. Feature/channel attribution: native torch (saliency/IG/smoothgrad/ablation) + optional backends.

Native methods need no extra deps. Captum (if installed), SHAP and LIME (installed here) are
available as alternate backends via lazy imports; missing backends raise informative errors.
Score function: softmin prototype distance (differentiable).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class SoftProtoScore(nn.Module):
    """Differentiable anomaly score for one window: softmin distance to prototypes."""

    def __init__(self, encoder: nn.Module, prototypes: np.ndarray, temp: float = 0.1):
        super().__init__()
        self.enc = encoder
        self.register_buffer("proto", torch.tensor(np.asarray(prototypes, dtype=np.float32)))
        self.temp = temp

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, C) -> (B,) scores
        z = self.enc(x)["rep"]
        D = ((z[:, None, :] - self.proto[None, :, :]) ** 2).sum(-1).sqrt()
        w = torch.softmax(-D / self.temp, dim=1)
        return (w * D).sum(1)


def saliency(score_fn: nn.Module, x: np.ndarray) -> np.ndarray:
    """|d score / dx| per (T, C)."""
    score_fn.eval()
    t = torch.tensor(x[None].astype(np.float32), requires_grad=True)
    score_fn(t).sum().backward()
    return t.grad[0].abs().detach().numpy().astype(np.float64)


def integrated_gradients(score_fn: nn.Module, x: np.ndarray, baseline: np.ndarray | None = None,
                         steps: int = 20) -> np.ndarray:
    """Path integral of grads from baseline (default zeros) to x. (T, C) out."""
    score_fn.eval()
    x = x.astype(np.float32)
    b = np.zeros_like(x) if baseline is None else baseline.astype(np.float32)
    total = torch.zeros_like(torch.tensor(x))
    for k in range(1, steps + 1):
        xk = torch.tensor(b + (x - b) * k / steps, requires_grad=True)
        score_fn(xk[None]).sum().backward()
        total = total + xk.grad.detach()
    return ((torch.tensor(x - b) * total / steps).abs().numpy()).astype(np.float64)


def smoothgrad(score_fn: nn.Module, x: np.ndarray, n: int = 10, sigma: float = 0.05,
               seed: int = 0) -> np.ndarray:
    """Noise-averaged saliency (GradientSHAP-style, dependency-free)."""
    rng = np.random.default_rng(seed)
    acc = np.zeros_like(x, dtype=np.float64)
    for _ in range(n):
        acc += saliency(score_fn, x + rng.normal(0, sigma, x.shape).astype(np.float32))
    return acc / n


def ablation_scores(score_fn: nn.Module, x: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Channel ablation: score drop when each channel is replaced by baseline. (C,) out."""
    score_fn.eval()
    with torch.no_grad():
        s0 = float(score_fn(torch.tensor(x[None].astype(np.float32))).item())
        drops = []
        for c in range(x.shape[1]):
            xc = x.copy()
            xc[:, c] = baseline[:, c] if baseline.ndim == 2 else baseline
            drops.append(s0 - float(score_fn(torch.tensor(xc[None].astype(np.float32))).item()))
    return np.array(drops, dtype=np.float64)


def captum_ig(score_fn: nn.Module, x: np.ndarray, steps: int = 20) -> np.ndarray:
    """Captum backend (optional)."""
    try:
        from captum.attr import IntegratedGradients
    except ImportError as e:
        raise RuntimeError("captum not installed; use native integrated_gradients") from e
    ig = IntegratedGradients(score_fn)
    t = torch.tensor(x[None].astype(np.float32))
    attr, _ = ig.attribute(t, torch.zeros_like(t), n_steps=steps, return_convergence_delta=True)
    return attr[0].abs().detach().numpy().astype(np.float64)


def lime_explain(score_fn: nn.Module, x: np.ndarray, n_samples: int = 500,
                 seed: int = 0) -> np.ndarray:
    """LIME-tabular backend over flattened (T*C) features. Returns |coef| reshaped (T, C)."""
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as e:
        raise RuntimeError("lime not installed") from e
    score_fn.eval()
    flat = x.astype(float).reshape(-1)
    bg = np.stack([flat, np.zeros_like(flat), flat * 0.5 + 0.1])
    exp = LimeTabularExplainer(bg, mode="regression", random_state=seed,
                               feature_names=[f"t{t}c{c}" for t in range(x.shape[0])
                                              for c in range(x.shape[1])])

    def _predict(M: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor(M.reshape(len(M), *x.shape).astype(np.float32))
            return score_fn(t).numpy()

    coefs = np.zeros_like(flat)
    li = exp.explain_instance(flat, _predict, num_features=len(flat))
    for feat, w in li.local_exp[0]:
        coefs[feat] = abs(w)
    return coefs.reshape(x.shape).astype(np.float64)


ATTRIBUTORS = {"saliency": saliency, "ig": integrated_gradients, "smoothgrad": smoothgrad}
