"""Bundle builder: train source-only contrastive on synthetic, save models/bundle/<name>."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from tsad.config import load_config
from tsad.seed import set_seed
from scripts.train import prepare_and_train
from tsad.api.registry import save_bundle
from experiments.common import ensure_defaults


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--opt", nargs="*", default=[])
    ap.add_argument("--name", default="synthetic_tcn")
    args = ap.parse_args()
    cfg = ensure_defaults(load_config(args.config, args.opt))
    set_seed(int(cfg.seed))
    art = prepare_and_train(cfg)
    out = save_bundle(art, cfg, Path("models") / "bundle" / args.name)
    print(f"bundle -> {out} (strict_f1={art['report']['strict_f1']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
