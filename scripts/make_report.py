"""Phase-10 report generator: figures (PDF/PNG) + tables (LaTeX/MD) + drafted sections.

RULE: prose references ONLY numbers read from actual run CSVs. Missing inputs -> 'NOT RUN' placeholders.
Usage: python scripts/make_report.py [--reports-dir reports --out reports/final]
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def collect(reports: Path, out: Path) -> dict[str, list[dict]]:
    """Gather every results CSV (recursive, e.g. explain galleries). Skips the output dir itself."""
    data: dict[str, list[dict]] = {}
    import logging

    log = logging.getLogger("report")
    for p in sorted(reports.rglob("*.csv")):
        if out in p.parents or p.parent == out:
            continue
        try:
            rows = read_csv(p)
            if rows:
                key = p.stem
                if key in data:  # disambiguate gallery copies
                    key = f"{p.parent.name}_{p.stem}"
                data[key] = rows
        except Exception as e:
            log.warning("skipping unreadable %s: %s", p, e)
            continue
    return data


def _f(v, nd: int = 3) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def f1_summary(rows: list[dict]) -> str:
    """One-line summary of strict_f1 values in rows, or NOT RUN."""
    vals = []
    for r in rows:
        if "strict_f1" in r:
            try:
                vals.append(float(r["strict_f1"]))
            except ValueError:
                pass
    if not vals:
        return "NOT RUN (no strict_f1 in results)"
    import numpy as np

    v = np.array(vals)
    return f"n={len(v)}, mean={v.mean():.3f}, max={v.max():.3f}, min={v.min():.3f}"


def cond_table(rows: list[dict]) -> str:
    """Markdown rows for condition-like tables (adapt/e3)."""
    lines = []
    for r in rows:
        name = r.get("condition", r.get("model", r.get("explainer", "?")))
        lines.append(
            f"| {name} | {_f(r.get('strict_f1', r.get('deletion_auc_mean', 'n/a')))} | "
            f"{_f(r.get('auc_roc', r.get('strict_auc_roc', 'n/a')))} |"
        )
    return "\n".join(lines)


def draft_results(data: dict[str, list[dict]]) -> str:
    L = ["# Results (auto-drafted from actual runs — verify before submitting)", ""]
    for key in ("e1_smoke", "baselines_smoke", "contrast_smoke"):
        if key in data:
            L.append(f"## Single-domain ({key}): {f1_summary(data[key])}")
    if "adapt_smoke" in data:
        L += [
            "",
            "## Transfer / negative transfer (adapt_smoke)",
            "| condition | strict_F1 | AUC-ROC |",
            "|---|---|---|",
            cond_table(data["adapt_smoke"]),
        ]
    elif "adapt_default" in data:
        L += ["", "## Transfer (adapt_default)", cond_table(data["adapt_default"])]
    else:
        L += ["", "## Transfer: NOT RUN (no adapt_*.csv found)"]
    if "e3_smoke" in data:
        L += ["", "## Alignment comparison (e3_smoke)", cond_table(data["e3_smoke"])]
    if "e4_smoke" in data:
        L += [
            "",
            "## Negative-transfer study (e4_smoke)",
            "| kind | condition | strict_F1 | dissimilarity |",
            "|---|---|---|---|",
        ]
        for r in data["e4_smoke"]:
            L.append(
                f"| {r.get('harmful_kind')} | {r.get('condition')} | "
                f"{_f(r.get('strict_f1'))} | {_f(r.get('dissimilarity'))} |"
            )
    mkey = next((k for k in data if k.endswith("_metrics") and "cf" not in k), None)
    if mkey is not None:
        m = data[mkey]
        L += ["", f"## Explainability ({mkey}): explainer fidelity (deletion-AUC)"]
        for r in m:
            L.append(
                f"- {r.get('explainer')}: deletion-AUC {_f(r.get('deletion_auc'))}, "
                f"P@k {_f(r.get('loc_p_at_k'))}"
            )
    else:
        L += ["", "## Explainability metrics: NOT RUN (run make_explanations first)"]
    L += ["", "> Placeholders marked NOT RUN must be replaced by real runs before any submission."]
    return "\n".join(L) + "\n"


def draft_ablation(data: dict[str, list[dict]]) -> str:
    L = ["# Ablation summary (auto-drafted)", ""]
    if "e5_smoke" in data:
        L += ["| ablation | strict_F1 |", "|---|---|"]
        for r in data["e5_smoke"]:
            L.append(f"| {r.get('ablation')} | {_f(r.get('strict_f1'))} |")
    else:
        L.append("Ablations: NOT RUN (no e5_*.csv).")
    if "e7_smoke" in data:
        L += ["", "## Sensitivity (alpha x tau)", "| alpha | tau | strict_F1 |", "|---|---|---|"]
        for r in data["e7_smoke"]:
            L.append(f"| {r.get('alpha')} | {r.get('tau')} | {_f(r.get('strict_f1'))} |")
    return "\n".join(L) + "\n"


LIMITATIONS = """# Limitations (must accompany any submission)

1. Window-pooled embeddings dilute isolated point spikes (max+mean concat mitigates, not solves).
2. Unsupervised-percentile operating point is conservative relative to ranking quality (AUC 0.9+).
3. Negative-transfer effects are seed-dependent on tiny synthetic data; paper claims require 5-seed
   full runs AND real SMD/SMAP numbers (currently synthetic-only for transfer).
4. Point-adjusted metrics are reported beside strict only; PA inflates and must never stand alone.
5. Range/event-wise F1 are simplified (Tatbul-style / overlap>=1); VUS/affiliation not implemented.
6. /adapt is prototype-refresh, not full retraining. Conditional alignment is a chamfer option only.
"""


def gen_summary_fig(data: dict[str, list[dict]], path: Path) -> Path | None:
    """Bar figure from e3/adapt condition tables (real numbers only). None if no data."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    src = data.get("e3_smoke") or data.get("adapt_smoke") or data.get("adapt_default")
    if not src:
        return None
    names = [str(r.get("condition", r.get("model", "?"))) for r in src]
    vals = [float(r.get("strict_f1", 0) or 0) for r in src]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, vals)
    ax.set_ylabel("strict_F1 (actual runs)")
    ax.set_title("Alignment comparison (smoke runs)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    import matplotlib.pyplot as plt2

    plt2.close(fig)
    return path


def build(reports_dir: str | Path = "reports", out: str | Path = "reports/final") -> dict:
    reports, out = Path(reports_dir), Path(out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "slides").mkdir(parents=True, exist_ok=True)
    data = collect(reports, out)
    # 1. tables: LaTeX versions of key CSVs
    from experiments.common import to_latex

    made_tables = []
    for key in (
        "e1_smoke",
        "e3_smoke",
        "e4_smoke",
        "adapt_smoke",
        "adapt_default",
        "baselines_smoke",
        "contrast_smoke",
    ):
        if key in data:
            (out / "tables" / f"{key}.tex").write_text(to_latex(data[key], key, f"tab:{key}"))
            made_tables.append(key)
    # 2. figures: copy existing PNGs + generated summary (PNG+PDF)
    figs = []
    for p in sorted(reports.glob("*.png")) + sorted((reports / "exp" / "figs").glob("*.png")):
        dst = out / p.name
        shutil.copy(p, dst)
        figs.append(dst.name)
    sf = gen_summary_fig(data, out / "summary_alignment.png")
    if sf is not None:
        figs.append(sf.name)
        figs.append(sf.with_suffix(".pdf").name)
        for f in ("adapt_mmd.png", "data_summary.png"):
            if (out / f).exists():
                shutil.copy(out / f, out / "slides" / f)
        shutil.copy(sf, out / "slides" / sf.name)
    # 3. drafted sections (real numbers only / NOT RUN)
    (out / "results_section.md").write_text(draft_results(data))
    (out / "ablation_summary.md").write_text(draft_ablation(data))
    (out / "limitations.md").write_text(LIMITATIONS)
    return {
        "tables": made_tables,
        "figures": figs,
        "sections": ["results_section.md", "ablation_summary.md", "limitations.md"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--opt", nargs="*", default=[])
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--out", default="reports/final")
    args = ap.parse_args()
    res = build(args.reports_dir, args.out)
    print(f"report -> {args.out}: {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
