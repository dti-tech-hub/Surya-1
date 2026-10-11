"""E6: explainability evaluation — run gallery once per seed, aggregate explainer metrics."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import numpy as np

from experiments.common import seed_cfg


def run(cfg, seeds, outdir, smoke: bool = True) -> list[dict]:
    outdir = Path(outdir)
    rows = []
    for s in seeds:
        seed_cfg(cfg, s)  # validates + seeds (gallery script reads --opt seed=)
        gdir = outdir / f"gallery_s{s}"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "make_explanations.py"),
                "--config",
                "configs/smoke.yaml" if smoke else "configs/default.yaml",
                "--opt",
                f"seed={s}",
                "--out",
                str(gdir),
            ],
            check=True,
        )
        with open(gdir / "_metrics.csv") as f:
            per = list(csv.DictReader(f))
        by_exp: dict[str, list[float]] = {}
        by_loc: dict[str, list[float]] = {}
        for r in per:
            try:
                by_exp.setdefault(r["explainer"], []).append(float(r["deletion_auc"]))
                if r["loc_auprc"] not in ("nan", "", "None"):
                    by_loc.setdefault(r["explainer"], []).append(float(r["loc_auprc"]))
            except ValueError:
                continue
        for exp, vals in by_exp.items():
            locs = by_loc.get(exp, [float("nan")])
            rows.append(
                {
                    "seed": s,
                    "explainer": exp,
                    "deletion_auc_mean": round(float(np.mean(vals)), 4),
                    "loc_auprc_mean": round(float(np.nanmean(locs)), 4),
                    "n_windows": len(vals),
                }
            )
    return rows
