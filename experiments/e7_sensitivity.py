"""E7: sensitivity — alpha x tau grid for adaptive-full (+ threshold rule in full mode)."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from experiments.common import seed_cfg
from scripts.adapt import prep, run_condition, CONDITIONS
from tsad.logging_utils import get_logger


def run(cfg, seeds, outdir, smoke: bool = True, epochs: int = 3) -> list[dict]:
    log = get_logger("e7")
    alphas = [0.1, 0.5] if smoke else [0.1, 0.5, 1.0]
    taus = [0.5, 1.0] if smoke else [0.5, 1.0, 2.0]
    rows = []
    for s in seeds:
        c = seed_cfg(cfg, s)
        if int(c.data.n_domains) < 3:
            c.data.n_domains = 3
        _, pack, src_ids, _, _ = prep(c, harmful=True, harmful_kind="noise")
        for a in alphas:
            for t in taus:
                acfg = dict(CONDITIONS["adaptive-full"])
                acfg.update({"alpha": a, "tau": t})
                r = run_condition(c, pack, src_ids, "domain_0", f"a{a}-t{t}", acfg, log, epochs)
                rows.append({"seed": s, "alpha": a, "tau": t, **r})
    return rows
