"""Phase-3 train: single-domain contrastive (source-only) -> prototype scoring -> results row.

Labels are UNUSED in training (self-supervised). Labels used only to pick NORMAL train windows
for the prototype bank (mirrors 'train assumed normal') and for final test metrics.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from tsad.config import load_config, save_config
from tsad.seed import set_seed
from tsad.logging_utils import get_logger
from tsad.tracking import Tracker
from tsad.data.synthetic import SyntheticDataset
from tsad.data.normalize import PerDomainNormalizer
from tsad.data.windows import sliding_windows, window_label
from tsad.lightning.contrast import ContrastModule, make_loaders
from tsad.models.prototype import PrototypeBank
from tsad.models.scoring import map_to_timestamps, smooth, hybrid
from tsad.eval.metrics import full_report
from tsad.eval.threshold import select_threshold, calibrate
from tsad.eval.report import write_table
from tsad.viz.embed import plot_embeddings


def run_single_domain(cfg, domain_id: str = "domain_0") -> dict:
    log = get_logger("train")
    ds = SyntheticDataset(int(cfg.data.n_domains), int(cfg.data.length), int(cfg.data.n_channels),
                          float(cfg.data.anomaly_ratio), int(cfg.seed)).load()
    d = ds[domain_id]
    norm = PerDomainNormalizer().fit(d.train_series)
    tr, va, te = (norm.transform(s) for s in (d.train_series, d.val_series, d.test_series))
    W, S = int(cfg.data.window), int(cfg.data.stride)
    tr_w = sliding_windows(tr, W, S)
    va_w = sliding_windows(va, W, S)
    te_w = sliding_windows(te, W, S)
    tr_wl = window_label(d.train_labels, W, S)
    te_starts = list(range(0, len(te) - W + 1, S))

    mod = ContrastModule(n_ch=tr.shape[1], encoder=str(cfg.model.encoder),
                         d=int(cfg.model.embedding_dim), tau=float(cfg.loss.tau),
                         w_nce=float(cfg.loss.w_nce), w_ts2vec=float(cfg.loss.w_ts2vec),
                         w_mask=float(cfg.loss.w_mask), lr=float(cfg.train.lr))
    tr_l, va_l = make_loaders(tr_w, va_w, list(cfg.augment or ["jitter"]),
                              int(cfg.train.batch_size), int(cfg.seed))
    ckpt = Path("checkpoints")
    ckpt.mkdir(exist_ok=True)
    trainer = L.Trainer(max_epochs=int(cfg.train.epochs), accelerator="auto",
                        enable_progress_bar=False, enable_model_summary=False,
                        gradient_clip_val=1.0,
                        precision="16-mixed" if __import__("torch").cuda.is_available() else 32,
                        callbacks=[EarlyStopping("val_loss", patience=3, mode="min"),
                                   ModelCheckpoint(str(ckpt), monitor="val_loss", mode="min")])
    trainer.fit(mod, tr_l, va_l)
    # prototype bank on NORMAL train windows only (train assumed normal)
    Ztr, _ = mod.embed(tr_w)
    clean = Ztr[tr_wl == 0]
    note = "clean-normal-windows"
    if len(clean) == 0:  # degenerate tiny splits: fall back to all windows (flagged, honest)
        clean, note = Ztr, "no-clean-windows-fallback"
        log.warning("no clean train windows; prototype bank fit on all train windows")
    bank = PrototypeBank(k=int(cfg.model.num_prototypes),
                         method=str(cfg.get("scoring", {}).get("method", "kmeans")),
                         seed=int(cfg.seed)).fit(clean)
    Zte, _ = mod.embed(te_w)
    te_win_scores = bank.score(Zte)
    Zva, _ = mod.embed(va_w)
    va_win_scores = bank.score(Zva)
    va_ts = smooth(map_to_timestamps(va_win_scores, list(range(0, len(va) - W + 1, S)),
                                     len(va), W), 5)
    va_normal = d.val_labels == 0  # unsupervised thresholds: NORMAL val timesteps only
    # Guard band: mapping/smoothing smears anomaly scores onto neighbours — exclude
    # timesteps within one window of any val anomaly so the reference stays clean.
    anom_idx = np.where(d.val_labels == 1)[0]
    if len(anom_idx):
        bad = np.zeros(len(va), bool)
        for a in anom_idx:
            bad[max(0, a - W):a + W + 1] = True
        va_normal = va_normal & ~bad
    te_scores = smooth(map_to_timestamps(te_win_scores, te_starts, len(te), W),
                       int(cfg.get("scoring", {}).get("smooth_width", 5)))
    hw = float(cfg.get("scoring", {}).get("hybrid_weight", 0.7))
    # Hybrid with reconstruction error when the masked-recon aux trained the decoder (w_mask>0)
    if float(cfg.loss.w_mask) > 0:
        te_re = smooth(map_to_timestamps(mod.window_recon_error(te_w).mean(axis=1), te_starts,
                                         len(te), W), 5)
        va_re = smooth(map_to_timestamps(mod.window_recon_error(va_w).mean(axis=1),
                                         list(range(0, len(va) - W + 1, S)), len(va), W), 5)
        te_scores = hybrid(te_scores, te_re, hw)
        va_ts = hybrid(va_ts, va_re, hw)
        note += "+hybrid-recon"
    va_ref = va_ts[va_normal] if va_normal.sum() else va_ts
    va_ref_z, mu, sd = calibrate(va_ref, va_ref)  # val-only calibration (no test leakage)
    te_scores = (te_scores - mu) / sd  # same val stats -> test (hybrid-aware: mu/sd fit post-hybrid)
    thr = select_threshold(str(cfg.threshold.method), va_ref_z, None, float(cfg.threshold.percentile))
    rep = full_report(d.test_labels, te_scores, thr)
    log.info(f"contrastive@{domain_id}: strict_f1={rep['strict_f1']:.3f} pa_f1={rep['pa_f1']:.3f}")
    plot_embeddings(Zte, window_label(d.test_labels, W, S),
                    f"reports/embed_{cfg.model.encoder}_{domain_id}.png",
                    f"{cfg.model.encoder} {domain_id}")
    row = {"model": f"contrast-{cfg.model.encoder}", "domain": domain_id, "proto_note": note,
           **{k: round(float(v), 4) for k, v in rep.items() if isinstance(v, (int, float))}}
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--opt", nargs="*", default=[])
    ap.add_argument("--out", default="reports/contrast")
    args = ap.parse_args()
    cfg = load_config(args.config, args.opt)
    set_seed(int(cfg.seed))
    # ensure Phase-3 defaults exist on older configs
    for k, v in {"encoder": "tcn", "embedding_dim": 8, "num_prototypes": 3}.items():
        cfg.model.setdefault(k, v)
    for k, v in {"tau": 0.1, "w_nce": 1.0, "w_ts2vec": 0.0, "w_mask": 0.0}.items():
        cfg.setdefault("loss", {}).setdefault(k, v) if "loss" in cfg else None
    if "loss" not in cfg:
        cfg.loss = {"tau": 0.1, "w_nce": 1.0, "w_ts2vec": 0.0, "w_mask": 0.0}
    if "augment" not in cfg:
        cfg.augment = ["jitter"]
    tr = Tracker(backend=str(cfg.tracking.backend), experiment=str(cfg.tracking.experiment))
    tr.start("contrast", cfg)
    row = run_single_domain(cfg)
    tr.log_metrics({k: v for k, v in row.items() if isinstance(v, float)}, step=0)
    tr.stop()
    write_table([row], args.out)
    save_config(cfg, "reports/last_train_config.yaml")
    print(f"wrote {args.out}.{{csv,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
