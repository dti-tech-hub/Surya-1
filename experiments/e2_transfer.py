"""E2: cross-domain transfer — source->target pairs + target-only ref + transfer gain."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from experiments.common import seed_cfg
from scripts.train import run_single_domain
from tsad.eval.transfer import transfer_report


def run(cfg, seeds, outdir, smoke: bool = True) -> list[dict]:
    pairs = [("domain_1", "domain_0"), ("domain_0", "domain_1")]
    rows = []
    for s in seeds:
        c = seed_cfg(cfg, s)
        refs = {}
        for tgt in ("domain_0", "domain_1"):
            r = run_single_domain(c, tgt, tgt)
            refs[tgt] = r["strict_f1"]
            rows.append({"seed": s, "pair": f"{tgt}->{tgt} (target-only)", **r})
        for src, tgt in pairs:
            r = run_single_domain(c, src, tgt)
            tr = transfer_report(refs[tgt], r["strict_f1"], refs[tgt])
            rows.append(
                {
                    "seed": s,
                    "pair": f"{src}->{tgt}",
                    **r,
                    "t_gain_vs_targetonly": round(tr["transfer_gain"], 4),
                    "t_neg": tr["negative_transfer"],
                }
            )
    return rows
