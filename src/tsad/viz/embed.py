"""Embedding visualization (t-SNE default; UMAP if installed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_embeddings(
    Z: np.ndarray, labels: np.ndarray, path: str | Path, title: str = "embeddings"
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    zn = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    try:
        import umap  # optional

        E = umap.UMAP(n_components=2, random_state=0).fit_transform(zn)
        method = "UMAP"
    except Exception:
        n = min(30, max(1, len(zn) - 1))
        E = TSNE(n_components=2, random_state=0, perplexity=n).fit_transform(zn)
        method = "t-SNE"
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(E[:, 0], E[:, 1], c=np.asarray(labels), cmap="coolwarm", s=8, alpha=0.7)
    ax.set_title(f"{title} ({method})")
    fig.colorbar(sc, ax=ax, label="anomalous window")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
