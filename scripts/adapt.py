"""Phase-4 adapt smoke: source->target transfer with a deliberately HARMFUL source.

Conditions: source-only(no-align) | uniform-MMD | adaptive-full — all trained on
[related + harmful] sources, evaluated on target test. transfer_report() flags negative transfer.
Writes reports/adapt_smoke.{csv,md} + reports/adapt_mmd.png (MMD before/after + F1 bars).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
import lightning as L
from tsad.config import load_config
from tsad.seed import set_seed
from tsad.logging_utils import get_logger
from tsad.data.synthetic import SyntheticDataset
from tsad.data.normalize import PerDomainNormalizer
from tsad.data.windows import sliding_windows, window_label
from tsad.lightning.adapt import AdaptModule, make_paired_loaders
from tsad.losses.mmd import mmd2
from tsad.models.prototype import PrototypeBank
from tsad.models.scoring import map_to_timestamps, smooth
from tsad.eval.metrics import full_report
from tsad.eval.threshold import select_threshold, calibrate
from tsad.eval.transfer import transfer_report
from tsad.eval.report import write_table


CONDITIONS = {
    "source-only": {"method": "none", "use_gate": False, "use_weights": False,
                    "use_monitor": False, "alpha": 0.0, "beta": 0.0, "gamma": 0.0,
                    "sources": "all"},
    "uniform-clean": {"method": "mmd", "use_gate": False, "use_weights": False,
                      "use_monitor": False, "alpha": 0.5, "beta": 0.0, "gamma": 0.0,
                      "sources": "related"},
    "uniform-mixed": {"method": "mmd", "use_gate": False, "use_weights": False,
                      "use_monitor": False, "alpha": 0.5, "beta": 0.0, "gamma": 0.0,
                      "sources": "all"},
    "adaptive-full": {"method": "mmd", "use_gate": True, "use_weights": True,
                      "use_monitor": True, "alpha": 0.5, "beta": 0.1, "gamma": 0.01,
                      "tau": 1.0, "sources": "all"},
}


def prep(cfg, target: str = "domain_0"):
    ds = SyntheticDataset(int(cfg.data.n_domains), int(cfg.data.length), int(cfg.data.n_channels),
                          float(cfg.data.anomaly_ratio), int(cfg.seed),
                          harmful_last=True, harmful_kind="noise").load()
    assert target in ds and len(ds) >= 3, "need >=3 domains incl. harmful last"
    src_ids = [i for i in sorted(ds) if i != target]
    norms = {i: PerDomainNormalizer().fit(ds[i].train_series) for i in ds}
    W, S = int(cfg.data.window), int(cfg.data.stride)
    pack = {}
    for i, d in ds.items():
        n = norms[i]
        pack[i] = {k: v for k, v in zip(
            ("tr", "va", "te", "trl", "val", "tel"),
            (*[n.transform(s) for s in (d.train_series, d.val_series, d.test_series)],
             d.train_labels, d.val_labels, d.test_labels))}
    return ds, pack, src_ids, W, S


def run_condition(cfg, pack, src_ids, target, name, acfg, log, epochs: int):
    W, S = int(cfg.data.window), int(cfg.data.stride)
    use_ids = [src_ids[0]] if acfg.get("sources") == "related" else src_ids  # related-only ref
    src_w = [sliding_windows(pack[i]["tr"], W, S) for i in use_ids]
    tgt_w = sliding_windows(pack[target]["tr"], W, S)
    src_v = [sliding_windows(pack[i]["va"], W, S) for i in use_ids]
    tgt_v = sliding_windows(pack[target]["va"], W, S)
    mod = AdaptModule(n_ch=pack[target]["tr"].shape[1], n_src=len(use_ids),
                      encoder=str(cfg.model.encoder), d=int(cfg.model.embedding_dim),
                      lr=float(cfg.train.lr), align_cfg=acfg)
    mod.set_domain_cache(src_w, tgt_w)  # stable input-space domain weights
    tr_l, va_l = make_paired_loaders(src_w, tgt_w, src_v, tgt_v,
                                     list(cfg.get("augment", ["jitter"])),
                                     int(cfg.train.batch_size), int(cfg.seed))
    t = L.Trainer(max_epochs=epochs, accelerator="auto",
                  enable_progress_bar=False, enable_model_summary=False, logger=False,
                  enable_checkpointing=False, gradient_clip_val=1.0)
    t.fit(mod, tr_l, va_l)
    # MMD before/after (raw train windows, source-concat vs target)
    import torch as _t

    with _t.no_grad():
        Sall = _t.tensor(np.concatenate(src_w), dtype=_t.float32).reshape(len(np.concatenate(src_w)), -1)
        Tall = _t.tensor(tgt_w, dtype=_t.float32).reshape(len(tgt_w), -1)
        mmd_before = float(mmd2(Sall, Tall))
        Zs = [mod.embed(s) for s in src_w]
        Zt = mod.embed(tgt_w)
        mall = _t.tensor(np.concatenate(Zs), dtype=_t.float32)
        mall_t = _t.tensor(Zt, dtype=_t.float32)
        mmd_after = float(mmd2(mall, mall_t))
    # score target test with prototype bank on NORMAL source-train embeddings (assumed normal)
    src_tr_labs = [window_label(pack[i]["trl"], W, S) for i in use_ids]
    Zclean = np.concatenate([z[l == 0] for z, l in zip(Zs, src_tr_labs)])
    bank = PrototypeBank(k=int(cfg.model.num_prototypes), seed=int(cfg.seed))
    bank.fit(Zclean if len(Zclean) else np.concatenate(Zs))
    te, tel = pack[target]["te"], pack[target]["tel"]
    te_w = sliding_windows(te, W, S)
    starts = list(range(0, len(te) - W + 1, S))
    te_ts = smooth(map_to_timestamps(bank.score(mod.embed(te_w)), starts, len(te), W), 5)
    va, val = pack[target]["va"], pack[target]["val"]
    va_w = sliding_windows(va, W, S)
    va_ts = smooth(map_to_timestamps(bank.score(mod.embed(va_w)),
                                     list(range(0, len(va) - W + 1, S)), len(va), W), 5)
    ok = val == 0
    ref = va_ts[ok] if ok.sum() else va_ts
    refz, mu, sd = calibrate(ref, ref)
    # Operating point: best-F1 on VAL (labels allowed for val; test labels never used).
    # Same rule for all conditions -> fair comparison of representation/alignment quality.
    thr = select_threshold("best-f1", va_ts, val)
    rep = full_report(tel, (te_ts - mu) / sd, (thr - mu) / sd)
    w = mod._last_weights
    wlist = [round(float(x), 3) for x in (w.cpu() if w is not None else [])]
    log.info(f"{name}: strict_f1={rep['strict_f1']:.3f} mmd {mmd_before:.3f}->{mmd_after:.3f} weights={wlist}")
    return {"condition": name, "strict_f1": round(rep["strict_f1"], 4),
            "auc_roc": round(rep["strict_auc_roc"], 4), "mmd_before": round(mmd_before, 4),
            "mmd_after": round(mmd_after, 4), "domain_weights": str(wlist)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--opt", nargs="*", default=[])
    ap.add_argument("--out", default="reports/adapt_smoke")
    ap.add_argument("--epochs", type=int, default=None, help="override cfg.train.epochs")
    args = ap.parse_args()
    cfg = load_config(args.config, args.opt)
    for k, v in {"encoder": "tcn", "embedding_dim": 8, "num_prototypes": 3}.items():
        cfg.model.setdefault(k, v)
    set_seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    log = get_logger("adapt")
    if int(cfg.data.n_domains) < 3:  # need related + harmful + target
        cfg.data.n_domains = 3
    ds, pack, src_ids, W, S = prep(cfg)
    log.info(f"sources={src_ids} (last=harmful) target=domain_0")
    epochs = args.epochs or int(cfg.train.epochs)
    rows = [run_condition(cfg, pack, src_ids, "domain_0", n, a, log, epochs)
            for n, a in CONDITIONS.items()]
    base = next(r["strict_f1"] for r in rows if r["condition"] == "source-only")
    for r in rows:
        if r["condition"] != "source-only":
            tr = transfer_report(base, r["strict_f1"])
            r.update({f"t_{k}": v for k, v in tr.items() if k != "source_only_f1"})
    write_table(rows, args.out)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar([r["condition"] for r in rows], [r["strict_f1"] for r in rows])
    ax[0].set_title("Target strict_F1 (harmful source present)")
    ax[0].tick_params(axis="x", rotation=15)
    x = np.arange(len(rows))
    ax[1].bar(x - 0.2, [r["mmd_before"] for r in rows], 0.4, label="MMD before")
    ax[1].bar(x + 0.2, [r["mmd_after"] for r in rows], 0.4, label="MMD after")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([r["condition"] for r in rows], rotation=15)
    ax[1].set_title("Source-target MMD before/after")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig("reports/adapt_mmd.png")
    print(f"wrote {args.out}.{{csv,md}} + reports/adapt_mmd.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
