"""Gallery builder: per-anomaly PNG (overlay + attribution heatmap) + index HTML."""
from __future__ import annotations
from pathlib import Path
import numpy as np


def anomaly_panel(observed: np.ndarray, medoid: np.ndarray, attr: np.ndarray,
                  path: str | Path, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(8, 5), gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(observed.mean(1), color="crimson", label="observed")
    ax[0].plot(medoid.mean(1), color="steelblue", label="nearest normal")
    ax[0].set_title(title, fontsize=9)
    ax[0].legend(fontsize=8)
    im = ax[1].imshow(np.abs(attr).T, aspect="auto", cmap="hot", origin="lower")
    ax[1].set_xlabel("timestep")
    ax[1].set_ylabel("channel")
    fig.colorbar(im, ax=ax[1], label="|attr|")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def index_html(items: list[dict], path: str | Path, title: str = "Explanation gallery") -> Path:
    rows = "\n".join(
        f"<div><h3>#{it['window_id']} t={it['t_start']}-{it['t_end']} score={it['score']:.3f}</h3>"
        f"<img src='{Path(it['panel']).name}' width='760'><p>{it['summary']}</p></div>"
        for it in items)
    path = Path(path)
    path.write_text(f"<html><head><title>{title}</title></head><body><h1>{title}</h1>{rows}</body></html>")
    return path
