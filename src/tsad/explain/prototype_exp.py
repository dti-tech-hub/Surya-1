"""1. Prototype-based explanation: nearest normal prototype, medoid window, overlay plot."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def prototype_part(
    bank, Z_win: np.ndarray, train_windows: np.ndarray, train_embs: np.ndarray, proto_k: int = 2
) -> dict:
    """Z_win: (d,) embedding of the flagged window. Returns nearest protos + medoid overlay data."""
    from tsad.explain.base import topk_channels  # noqa (re-export convenience)

    D = np.linalg.norm(bank.prototypes - Z_win[None, :], axis=1)
    order = np.argsort(D)[:proto_k]
    # medoid: train window closest to nearest prototype
    d_to_p0 = np.linalg.norm(train_embs - bank.prototypes[order[0]][None, :], axis=1)
    med = train_windows[int(np.argmin(d_to_p0))]
    return {
        "nearest": [{"proto_id": int(i), "distance": float(D[i])} for i in order],
        "medoid_window": med.astype(float).tolist(),
        "n_prototypes": len(bank.prototypes),
    }


def overlay_plot(
    observed: np.ndarray,
    medoid: np.ndarray,
    path: str | Path,
    title: str = "observed vs nearest normal",
) -> Path:
    """observed/medoid: (T, C). Overlays channel means + shades max-deviation channel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observed, medoid = np.asarray(observed), np.asarray(medoid)
    dev = np.abs(observed - medoid).mean(axis=0)
    c_star = int(np.argmax(dev))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(observed.mean(axis=1), label="observed (ch-mean)", color="crimson")
    ax.plot(medoid.mean(axis=1), label="nearest normal (ch-mean)", color="steelblue")
    ax.plot(
        observed[:, c_star],
        label=f"observed ch{c_star}",
        color="crimson",
        linestyle="--",
        alpha=0.7,
    )
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    import matplotlib.pyplot as plt2

    plt2.close(fig)
    return path
