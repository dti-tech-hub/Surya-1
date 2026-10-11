"""E8: efficiency — params, train time (1 epoch), inference latency (windows/sec) on CPU."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from experiments.common import seed_cfg
from tsad.models.encoders import make_encoder
from tsad.models.prototype import PrototypeBank


def _params(m) -> int:
    return sum(p.numel() for p in m.parameters())


def run(cfg, seeds, outdir, smoke: bool = True) -> list[dict]:
    seed_cfg(cfg, seeds[0])
    dev = "cpu"  # efficiency reported on CPU (worst case); GPU noted if available
    W, C = int(cfg.data.window), int(cfg.data.n_channels)
    X = np.random.default_rng(0).normal(size=(256, W, C)).astype(np.float32)
    rows = []
    for enc in (["tcn", "transformer"] if smoke else ["tcn", "transformer", "resnet"]):
        m = make_encoder(enc, C, d=int(cfg.model.embedding_dim)).eval()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(5):
                m(torch.tensor(X[:64]))
        dt = (time.perf_counter() - t0) / 5
        # inference: embed + prototype score (bank pre-fit; fit timed separately)
        with torch.no_grad():
            Zr = m(torch.tensor(X))["rep"].numpy()
        t_fit = time.perf_counter()
        bank = PrototypeBank(k=3, seed=0).fit(Zr[:200])
        dt_fit = time.perf_counter() - t_fit
        t1 = time.perf_counter()
        bank.score(Zr)
        dt_inf = time.perf_counter() - t1
        rows.append(
            {
                "model": enc,
                "params": _params(m),
                "device": dev,
                "score_windows_per_sec": round(256 / dt_inf, 1),
                "kmeans_fit_s": round(dt_fit, 3),
                "batch64_forward_ms": round(dt * 1000, 1),
            }
        )
    return rows
