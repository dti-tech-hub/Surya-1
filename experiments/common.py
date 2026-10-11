"""Shared experiment helpers: seeding, table IO (CSV/MD/LaTeX), stats, plots."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from tsad.eval.report import write_table
from tsad.seed import set_seed


def ensure_defaults(cfg):
    """Fill Phase-3+ defaults so smoke.yaml (minimal) works in every experiment."""
    for k, v in {"encoder": "tcn", "embedding_dim": 8, "num_prototypes": 3}.items():
        cfg.model.setdefault(k, v)
    if "loss" not in cfg:
        cfg.loss = {"tau": 0.1, "w_nce": 1.0, "w_ts2vec": 0.0, "w_mask": 0.0}
    if "augment" not in cfg:
        cfg.augment = ["jitter"]
    return cfg


def seed_cfg(cfg, seed: int):
    c = deepcopy(cfg)
    c.seed = seed
    set_seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
    return c


def save_all(rows: list[dict], out_prefix: str | Path) -> dict:
    """CSV + markdown (via write_table) + LaTeX booktabs. Returns paths."""
    cp, mp = write_table(rows, out_prefix)
    tp = Path(str(out_prefix) + ".tex")
    tp.write_text(to_latex(rows))
    return {"csv": str(cp), "md": str(mp), "tex": str(tp)}


def to_latex(
    rows: list[dict], caption: str = "Results (mean values)", label: str = "tab:res"
) -> str:
    if not rows:
        return "% empty table\n"
    keys = sorted({k for r in rows for k in r})
    head = " & ".join(_tex(k) for k in keys) + r" \\"
    body = "\n".join(" & ".join(_tex_cell(r.get(k, "")) for k in keys) + r" \\" for r in rows)
    return (
        "\\begin{table}[h]\n\\centering\n\\caption{"
        + caption
        + "}\n\\label{"
        + label
        + "}\n"
        + "\\begin{tabular}{"
        + "l" * len(keys)
        + "}\n\\toprule\n"
        + head
        + "\n\\midrule\n"
        + body
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def _tex(s: str) -> str:
    return str(s).replace("_", "\\_")


def _tex_cell(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return _tex(v)


def paired_stats(a: list[float], b: list[float]) -> dict:
    """Paired t-test + Wilcoxon. Returns n/a strings when seeds < 2 or scipy missing."""
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return {"t_p": "n/a (need >=2 seeds)", "w_p": "n/a (need >=2 seeds)"}
    try:
        from scipy import stats

        return {
            "t_p": round(float(stats.ttest_rel(a, b).pvalue), 4),
            "w_p": round(float(stats.wilcoxon(a, b).pvalue), 4),
        }
    except Exception as e:
        return {"t_p": f"n/a ({e})", "w_p": "n/a"}


def mean_std_table(
    per_seed: list[dict], group_keys: tuple[str, ...] = ("model",), metric: str = "strict_f1"
) -> list[dict]:
    """Aggregate per-seed rows into mean±std rows grouped by group_keys."""
    groups: dict[tuple, list[float]] = {}
    for r in per_seed:
        groups.setdefault(tuple(r[k] for k in group_keys), []).append(float(r[metric]))
    out = []
    for key, vals in sorted(groups.items()):
        v = np.array(vals)
        row = {k: key[i] for i, k in enumerate(group_keys)}
        row.update(
            {
                metric + "_mean": round(float(v.mean()), 4),
                metric + "_std": round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4),
                "n_seeds": len(v),
            }
        )
        out.append(row)
    return out


def barplot(
    cats: list[str],
    series: dict[str, list[float]],
    path: str | Path,
    title: str = "",
    ylabel: str = "strict_F1",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(cats))
    w = 0.8 / max(1, len(series))
    fig, ax = plt.subplots(figsize=(max(6, len(cats) * 1.2), 4))
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + i * w, vals, w, label=name)
    ax.set_xticks(x + w * (len(series) - 1) / 2)
    ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def heatmap(
    mat: np.ndarray, xs: list[str], ys: list[str], path: str | Path, title: str = ""
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(5, len(xs)), max(4, len(ys) * 0.8)))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(xs)), xs)
    ax.set_yticks(range(len(ys)), ys)
    for i in range(len(ys)):
        for j in range(len(xs)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
