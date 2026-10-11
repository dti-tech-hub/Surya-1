"""Save synthetic domains to data/synthetic/*.npz (for inspection/reuse)."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsad.config import load_config
from tsad.seed import set_seed
from tsad.data.synthetic import SyntheticDataset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--opt", nargs="*", default=[])
    ap.add_argument("--out", default="data/synthetic")
    args = ap.parse_args()
    cfg = load_config(args.config, args.opt)
    set_seed(int(cfg.seed))
    ds = SyntheticDataset(int(cfg.data.n_domains), int(cfg.data.length),
                          int(cfg.data.n_channels), float(cfg.data.anomaly_ratio), int(cfg.seed))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import numpy as np

    for did, d in ds.load().items():
        np.savez(out / f"{did}.npz", train=d.train_series, val=d.val_series, test=d.test_series,
                 test_labels=d.test_labels, test_mask=d.test_channel_mask)
    print(f"saved {len(ds.load())} domains -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
