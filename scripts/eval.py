"""Phase-2 eval: run all baselines on synthetic (single-domain) -> results CSV+markdown.

Thresholds fit on VAL only (percentile default, CLI-overridable). Reports strict + PA (flagged).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from tsad.config import load_config
from tsad.seed import set_seed
from tsad.logging_utils import get_logger
from tsad.data.synthetic import SyntheticDataset
from tsad.data.normalize import PerDomainNormalizer
from tsad.baselines import all_baselines
from tsad.eval.metrics import full_report
from tsad.eval.threshold import select_threshold, calibrate
from tsad.eval.report import write_table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--opt", nargs="*", default=[])
    ap.add_argument("--out", default="reports/baselines")
    args = ap.parse_args()
    cfg = load_config(args.config, args.opt)
    set_seed(int(cfg.seed))
    log = get_logger("eval")
    ds = SyntheticDataset(int(cfg.data.n_domains), int(cfg.data.length), int(cfg.data.n_channels),
                          float(cfg.data.anomaly_ratio), int(cfg.seed)).load()
    rows = []
    W = int(cfg.data.window)
    for did, d in ds.items():
        norm = PerDomainNormalizer().fit(d.train_series)
        tr, va, te = norm.transform(d.train_series), norm.transform(d.val_series), norm.transform(d.test_series)
        for bl in all_baselines(window=int(cfg.data.window), epochs=1, seed=int(cfg.seed)):
            bl.fit(tr)
            va_s, te_s = bl.score(va), bl.score(te)
            va_normal = d.val_labels == 0  # NORMAL val timesteps only (spec: percentile on normal val)
            anom_idx = np.where(d.val_labels == 1)[0]
            if len(anom_idx):  # guard band against score smearing near val anomalies
                bad = np.zeros(len(va), bool)
                for a in anom_idx:
                    bad[max(0, a - W):a + W + 1] = True
                va_normal = va_normal & ~bad
            va_ref = va_s[va_normal] if va_normal.sum() else va_s
            va_ref_z, mu, sd = calibrate(va_ref, va_ref)  # val-only calibration
            te_s = (te_s - mu) / sd
            thr = select_threshold(str(cfg.threshold.method), va_ref_z, None,
                                   float(cfg.threshold.percentile))
            rep = full_report(d.test_labels, te_s, thr)
            rows.append({"model": bl.name, "domain": did, **{k: round(float(v), 4) for k, v in rep.items()
                                                             if isinstance(v, (int, float))}})
            log.info(f"{bl.name}@{did}: strict_f1={rep['strict_f1']:.3f} pa_f1={rep['pa_f1']:.3f} (thr={thr:.3f})")
    cp, mp = write_table(rows, args.out)
    print(f"wrote {cp} + {mp} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
