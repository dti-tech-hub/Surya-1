"""E3: alignment comparison — none vs uniform MMD/CORAL/DANN vs adaptive-full."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from experiments.common import seed_cfg
from scripts.adapt import CONDITIONS, prep, run_condition
from tsad.logging_utils import get_logger

EXTRA = {
    "uniform-coral": {
        "method": "coral",
        "use_gate": False,
        "use_weights": False,
        "use_monitor": False,
        "alpha": 0.5,
        "beta": 0.0,
        "gamma": 0.0,
        "sources": "all",
    },
    "uniform-dann": {
        "method": "dann",
        "use_gate": False,
        "use_weights": False,
        "use_monitor": False,
        "alpha": 0.5,
        "beta": 0.0,
        "gamma": 0.0,
        "sources": "all",
    },
}


def run(cfg, seeds, outdir, smoke: bool = True, epochs: int = 3) -> list[dict]:
    log = get_logger("e3")
    conds = dict(CONDITIONS)
    if not smoke:
        conds.update(EXTRA)
    rows = []
    for s in seeds:
        c = seed_cfg(cfg, s)
        if int(c.data.n_domains) < 3:
            c.data.n_domains = 3
        _, pack, src_ids, _, _ = prep(c, harmful=True, harmful_kind="noise")
        for name, acfg in conds.items():
            r = run_condition(c, pack, src_ids, "domain_0", name, acfg, log, epochs)
            rows.append({"seed": s, **r})
    return rows
