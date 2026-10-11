"""Explanation-quality metrics: fidelity, stability, localization, counterfactual, comparison table."""

from __future__ import annotations

import numpy as np


def _rank(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x.ravel())).reshape(x.shape).astype(float)


def deletion_curve(
    score_fn, x: np.ndarray, attr: np.ndarray, baseline: np.ndarray, steps: int = 10
) -> np.ndarray:
    """Progressively replace top-attributed entries with baseline; record score. Higher drop = faithful."""
    import torch

    score_fn.eval()
    order = np.argsort(-np.abs(attr).ravel())
    chunks = np.array_split(order, steps)
    curve = []
    with torch.no_grad():
        xc = x.copy().ravel()
        base = (
            np.asarray(baseline).ravel()
            if np.asarray(baseline).size > 1
            else np.full_like(xc, baseline)
        )
        curve.append(float(score_fn(torch.tensor(x[None].astype(np.float32))).item()))
        for ch in chunks:
            xc[ch] = base[ch]
            curve.append(
                float(score_fn(torch.tensor(xc.reshape(x.shape)[None].astype(np.float32))).item())
            )
    return np.array(curve)


def deletion_auc(score_fn, x, attr, baseline, steps: int = 10) -> float:
    c = deletion_curve(score_fn, x, attr, baseline, steps)
    return float((c[0] - c).mean() / max(1e-9, abs(c[0])))  # normalized mean drop


def sufficiency_comprehensiveness(score_fn, x, attr, baseline, frac: float = 0.2) -> dict:
    """Keep top-frac (sufficiency: score kept) vs remove top-frac (comprehensiveness: score drop)."""
    import torch

    score_fn.eval()
    k = max(1, int(attr.size * frac))
    top = np.argsort(-np.abs(attr).ravel())[:k]
    with torch.no_grad():
        s0 = float(score_fn(torch.tensor(x[None].astype(np.float32))).item())
        b = np.asarray(baseline, dtype=float)
        b = b if b.shape == x.shape else np.zeros_like(x)
        keep = b.copy().ravel()
        keep[top] = x.ravel()[top]
        s_keep = float(
            score_fn(torch.tensor(keep.reshape(x.shape)[None].astype(np.float32))).item()
        )
        rem = x.copy().ravel()
        rem[top] = b.ravel()[top]
        s_rem = float(score_fn(torch.tensor(rem.reshape(x.shape)[None].astype(np.float32))).item())
    return {
        "sufficiency": float(s_keep / max(1e-9, abs(s0))),
        "comprehensiveness": float((s0 - s_rem) / max(1e-9, abs(s0))),
    }


def stability(attr: np.ndarray, attr_noisy: list[np.ndarray], k: int = 5) -> dict:
    """Rank correlation (Spearman approx via Pearson on ranks) + top-k Jaccard across perturbations."""
    r0 = _rank(np.abs(attr)).ravel()
    cors, jacs = [], []
    t0 = set(np.argsort(-np.abs(attr).ravel())[:k])
    for a in attr_noisy:
        r1 = _rank(np.abs(a)).ravel()
        cors.append(float(np.corrcoef(r0, r1)[0, 1]))
        t1 = set(np.argsort(-np.abs(a).ravel())[:k])
        jacs.append(float(len(t0 & t1) / max(1, len(t0 | t1))))
    return {"rank_corr_mean": float(np.mean(cors)), "topk_jaccard_mean": float(np.mean(jacs))}


def localization(attr: np.ndarray, mask: np.ndarray, k: int | None = None) -> dict:
    """mask: (T, C) GT anomalous entries. precision@k + AUPRC over flattened attributions."""
    from sklearn.metrics import average_precision_score

    a, m = np.abs(attr).ravel(), (np.asarray(mask) > 0).ravel()
    k = k or max(1, int(m.sum()))
    top = np.argsort(-a)[:k]
    out = {"precision_at_k": float(m[top].mean()) if k else 0.0, "k": int(k)}
    try:
        out["auprc"] = (
            float(average_precision_score(m, a)) if m.sum() and not m.all() else float("nan")
        )
    except Exception:
        out["auprc"] = float("nan")
    return out


def counterfactual_metrics(cf: dict) -> dict:
    return {
        "valid": bool(cf["valid"]),
        "sparsity": float(1 - cf["frac_changed"]),
        "proximity_l1": float(cf["delta_l1"]),
        "proximity_l2": float(cf["delta_l2"]),
    }
