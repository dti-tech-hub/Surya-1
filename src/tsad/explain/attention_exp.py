"""2. Attention-based explanation: (T,) weights -> heatmap + top timesteps."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def attention_part(attn: np.ndarray | None) -> dict:
    """attn: (T,) from Transformer/AttnPool (None for TCN/ResNet -> flagged unavailable)."""
    if attn is None:
        return {"available": False, "reason": "encoder has no attention (TCN/ResNet)"}
    a = np.asarray(attn, dtype=float).ravel()
    a = a / a.sum().clip(min=1e-12)
    return {
        "available": True,
        "weights": a.tolist(),
        "top_timesteps": [int(i) for i in np.argsort(-a)[:5]],
    }


def attention_heatmap(attn: np.ndarray, path: str | Path, title: str = "attention") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 1.6))
    ax.imshow(np.asarray(attn).reshape(1, -1), aspect="auto", cmap="hot")
    ax.set_title(title)
    ax.set_xlabel("timestep")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
