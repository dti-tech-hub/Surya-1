"""Stub entry points for later phases (Phase 0: parse config, no heavy deps)."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsad.config import load_config
from tsad.seed import set_seed
from tsad.logging_utils import get_logger


def _run(name: str):
    ap = argparse.ArgumentParser(prog=name)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--opt", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.opt)
    set_seed(int(cfg.seed))
    get_logger(name).info(f"{name}: loaded {args.config} (full impl lands in later phases)")
    print(f"{name.upper()} STUB OK")
    return 0
