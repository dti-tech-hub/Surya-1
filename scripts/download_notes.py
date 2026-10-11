"""Download helpers (real URLs where public; manual steps where access-required)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

NOTES = {
    "smd": (
        "https://github.com/NetManAIOps/OmniAnomaly",
        "data/smd/",
        "28 machines x 38ch. Place machine-{i}-{train,test,test_label}.npy under data/smd/. No checksum published; verify 38 channels on load.",
    ),
    "smap_msl": (
        "https://github.com/khundrakpam/telemanom",
        "data/smap_msl/",
        "Mirror of NASA SMAP/MSL. Layout data/smap_msl/{train,test,test_label}/*.npy.",
    ),
    "ucr": (
        "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/",
        "data/ucr/",
        "Convert each UCR anomaly series to <group>__<name>.npz with arrays series(T,), labels(T,).",
    ),
    "swat": (
        "https://itrust.sutd.edu.sg/itrust-labs_datasets/",
        "data/swat/",
        "ACCESS-REQUIRED: request iTrust approval, download SWaT normal+attack xlsx.",
    ),
    "yahoo": (
        "https://webscope.sandbox.yahoo.com",
        "data/yahoo/",
        "ACCESS-REQUIRED: Yahoo Webscope L6, A1Benchmark CSVs.",
    ),
}


def main() -> int:
    print("Dataset download notes (no fake data is ever generated for real loaders):\n")
    for k, (url, dest, note) in NOTES.items():
        print(f"[{k}]\n  url: {url}\n  dest: {dest}\n  note: {note}\n")
    print("Synthetic fallback: python scripts/make_synthetic.py --config configs/smoke.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
