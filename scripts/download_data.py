"""Per-dataset download stubs sharing notes (each prints its own instructions)."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsad.data.datasets import SMDDataset, SMAPMSLDataset, UCRDataset, SWaTDataset, YahooDataset  # noqa

MAP = {"smd": SMDDataset, "smap_msl": SMAPMSLDataset, "ucr": UCRDataset,
       "swat": SWaTDataset, "yahoo": YahooDataset}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="smd", choices=sorted(MAP))
    args = ap.parse_args()
    try:
        ds = MAP[args.dataset]()
        domains = ds.load()
        print(f"{args.dataset}: {len(domains)} domains present.")
    except Exception as e:
        print(str(e))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
