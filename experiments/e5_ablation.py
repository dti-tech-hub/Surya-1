"""E5: ablations — encoder type + alignment components + window/k ablations (smoke subset)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from omegaconf import OmegaConf

from experiments.common import seed_cfg
from scripts.adapt import CONDITIONS, prep, run_condition
from scripts.train import run_single_domain
from tsad.logging_utils import get_logger

ALIGN_ABL = {
    "full": {},
    "no-gate": {"use_gate": False},
    "no-weights": {"use_weights": False},
    "no-preserve": {"beta": 0.0},
    "no-monitor": {"use_monitor": False},
}


def run(cfg, seeds, outdir, smoke: bool = True, epochs: int = 3) -> list[dict]:
    log = get_logger("e5")
    rows = []
    encoders = ["tcn", "transformer"] if smoke else ["tcn", "transformer", "resnet"]
    for s in seeds:
        c = seed_cfg(cfg, s)
        for enc in encoders:  # encoder ablation (single-domain contrastive)
            ce = OmegaConf.merge(c, {"model": {"encoder": enc}})
            r = run_single_domain(ce, "domain_0")
            rows.append({"seed": s, "ablation": f"enc-{enc}", **r})
        # alignment-component ablation (adaptive-full toggles)
        if int(c.data.n_domains) < 3:
            c.data.n_domains = 3
        _, pack, src_ids, _, _ = prep(c, harmful=True, harmful_kind="noise")
        names = ["full", "no-gate"] if smoke else list(ALIGN_ABL)
        for name in names:
            acfg = dict(CONDITIONS["adaptive-full"])
            acfg.update(ALIGN_ABL[name])
            r = run_condition(c, pack, src_ids, "domain_0", f"align-{name}", acfg, log, epochs)
            rows.append({"seed": s, "ablation": f"align-{name}", **r})
    return rows
