"""E4: negative-transfer study — harmful-kind {clean, shift, noise} x method; perf vs dissimilarity."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import torch

from experiments.common import seed_cfg
from scripts.adapt import CONDITIONS, prep, run_condition
from tsad.data.windows import sliding_windows
from tsad.logging_utils import get_logger
from tsad.losses.mmd import mmd2


def _dissim(pack, src_ids, target, W, S) -> float:
    """Input-space MMD(harmful source, target) — the x-axis of the perf-vs-dissimilarity plot."""
    last = src_ids[-1]
    A = sliding_windows(pack[last]["tr"], W, S).reshape(-1, W * pack[last]["tr"].shape[1])
    B = sliding_windows(pack[target]["tr"], W, S).reshape(-1, W * pack[target]["tr"].shape[1])
    with torch.no_grad():
        return float(
            mmd2(torch.tensor(A, dtype=torch.float32), torch.tensor(B, dtype=torch.float32))
        )


def run(cfg, seeds, outdir, smoke: bool = True, epochs: int = 3) -> list[dict]:
    log = get_logger("e4")
    kinds = ["clean", "noise"] if smoke else ["clean", "shift", "noise"]
    methods = (
        ["uniform-mixed", "adaptive-full"]
        if smoke
        else ["source-only", "uniform-mixed", "adaptive-full"]
    )
    rows = []
    for s in seeds:
        c = seed_cfg(cfg, s)
        if int(c.data.n_domains) < 3:
            c.data.n_domains = 3
        for kind in kinds:
            harmful = kind != "clean"
            _, pack, src_ids, W, S = prep(
                c, harmful=harmful, harmful_kind="noise" if kind == "noise" else "shift"
            )
            dis = 0.0 if kind == "clean" else _dissim(pack, src_ids, "domain_0", W, S)
            for m in methods:
                r = run_condition(c, pack, src_ids, "domain_0", m, CONDITIONS[m], log, epochs)
                rows.append({"seed": s, "harmful_kind": kind, "dissimilarity": round(dis, 4), **r})
    return rows
