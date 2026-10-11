"""Phase-0 smoke: validates config+seed+logging+tracking pipeline on CPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsad.config import load_config
from tsad.logging_utils import get_logger
from tsad.seed import set_seed
from tsad.tracking import Tracker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--opt", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.opt)
    set_seed(int(cfg.seed))
    log = get_logger("smoke")
    log.info(f"smoke config: data={cfg.data.name} seed={cfg.seed}")
    tr = Tracker(backend=str(cfg.tracking.backend), experiment=str(cfg.tracking.experiment))
    tr.start("smoke", cfg)
    tr.log_metrics({"smoke_ok": 1.0}, step=0)
    tr.stop()
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
