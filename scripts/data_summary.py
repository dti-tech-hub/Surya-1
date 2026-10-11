"""Data-summary report: domains, lengths, anomaly ratio, per-domain stats + domain-shift plot.

Writes reports/data_summary.{csv,md,png}. Tries synthetic always; tries SMD/SMAP if present (skips gracefully).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from tsad.config import load_config
from tsad.seed import set_seed
from tsad.data.synthetic import SyntheticDataset
from tsad.data.registry import DomainRegistry
from tsad.data.normalize import PerDomainNormalizer


def _try_load(name: str):
    try:
        if name == "synthetic":
            return None  # handled by caller with cfg
        from tsad.data.datasets import SMDDataset, SMAPMSLDataset

        cls = {"smd": SMDDataset, "smap_msl": SMAPMSLDataset}[name]
        return cls().load()
    except Exception as e:
        print(f"[summary] {name}: skipped ({str(e).splitlines()[0] if str(e) else 'missing'})")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--opt", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.opt)
    set_seed(int(cfg.seed))
    ds = SyntheticDataset(int(cfg.data.n_domains), int(cfg.data.length), int(cfg.data.n_channels),
                          float(cfg.data.anomaly_ratio), int(cfg.seed))
    domains = ds.load()
    for extra in ("smd", "smap_msl"):
        got = _try_load(extra)
        if got:
            print(f"[summary] {extra}: {len(got)} domains found (stats only, not merged)")
    reg = DomainRegistry(domains)
    rows = reg.summary()
    rep = Path("reports")
    rep.mkdir(parents=True, exist_ok=True)
    import csv

    with open(rep / "data_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["domain", "train_len", "test_len", "channels",
                                          "train_mean", "train_std", "test_mean", "test_std",
                                          "anomaly_ratio_test"])
        w.writeheader()
        for r in rows:
            w.writerow({"domain": r["domain"], "train_len": r["train"]["len"], "test_len": r["test"]["len"],
                        "channels": r["train"]["channels"], "train_mean": round(r["train"]["mean"], 4),
                        "train_std": round(r["train"]["std"], 4), "test_mean": round(r["test"]["mean"], 4),
                        "test_std": round(r["test"]["std"], 4),
                        "anomaly_ratio_test": round(r["anomaly_ratio_test"], 4)})
    # per-domain normalization sanity (fit train only) + shift plot (test means per domain/channel)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = PerDomainNormalizer()
    for did, d in domains.items():
        n = norm.fit(d.train_series).transform(d.test_series)
        assert np.isfinite(n).all(), f"non-finite normalized {did}"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar([r["domain"] for r in rows], [r["anomaly_ratio_test"] for r in rows])
    axes[0].set_title("Test anomaly ratio per domain")
    axes[1].bar([r["domain"] for r in rows], [r["test"]["mean"] for r in rows])
    axes[1].set_title("Test mean per domain (shift indicator)")
    fig.tight_layout()
    fig.savefig(rep / "data_summary.png")
    md = ["# Data summary", "", f"domains: {len(rows)}",
          "", "| domain | train_len | test_len | ch | tr_mean | tr_std | te_mean | te_std | anom_ratio |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['domain']} | {r['train']['len']} | {r['test']['len']} | {r['train']['channels']} | "
                  f"{r['train']['mean']:.3f} | {r['train']['std']:.3f} | {r['test']['mean']:.3f} | "
                  f"{r['test']['std']:.3f} | {r['anomaly_ratio_test']:.4f} |")
    (rep / "data_summary.md").write_text("\n".join(md) + "\n")
    print(f"wrote reports/data_summary.{{csv,md,png}} ({len(rows)} domains)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
