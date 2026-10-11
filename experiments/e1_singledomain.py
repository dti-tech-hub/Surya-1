"""E1: single-domain benchmark — baselines vs contrastive (no adaptation)."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from experiments.common import seed_cfg
from scripts.eval import evaluate_baselines
from scripts.train import run_single_domain


def run(cfg, seeds, outdir, smoke: bool = True) -> list[dict]:
    rows = []
    for s in seeds:
        c = seed_cfg(cfg, s)
        for r in evaluate_baselines(c, domains=["domain_0"]):
            rows.append({"seed": s, **r})
        rows.append({"seed": s, **run_single_domain(c, "domain_0")})
    return rows
