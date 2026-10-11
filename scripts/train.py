"""Phase-3 train: single-domain contrastive (source-only) -> prototype scoring -> results row.

Labels are UNUSED in training (self-supervised). Labels used only to pick NORMAL train windows
for the prototype bank (mirrors 'train assumed normal') and for final test metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lightning as L
import numpy as np
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from tsad.config import load_config, save_config
from tsad.data.normalize import PerDomainNormalizer
from tsad.data.synthetic import SyntheticDataset
from tsad.data.windows import sliding_windows, window_label
from tsad.eval.metrics import full_report
from tsad.eval.report import write_table
from tsad.eval.threshold import calibrate, select_threshold
from tsad.lightning.contrast import ContrastModule, make_loaders
from tsad.logging_utils import get_logger
from tsad.models.prototype import PrototypeBank
from tsad.models.scoring import hybrid, map_to_timestamps, smooth
from tsad.seed import set_seed
from tsad.tracking import Tracker
from tsad.viz.embed import plot_embeddings


def prepare_and_train(cfg, domain_id: str = "domain_0", test_domain_id: str | None = None) -> dict:
    """Train contrastive model + build scoring artifacts. Shared by train.py and make_explanations.py.

    domain_id: training (source) domain. test_domain_id: eval domain (None = same domain).
    Cross-domain: per-domain norms (each fit on own train), threshold from SOURCE val
    (no target labels at all — strictest transfer protocol).
    Returns dict with mod, bank, W, S, te/te_w/tel/starts, te_scores, thr, Zte, d, note.
    """
    log = get_logger("train")
    ds = SyntheticDataset(
        int(cfg.data.n_domains),
        int(cfg.data.length),
        int(cfg.data.n_channels),
        float(cfg.data.anomaly_ratio),
        int(cfg.seed),
    ).load()
    test_domain_id = test_domain_id or domain_id
    dtr, d = ds[domain_id], ds[test_domain_id]
    ntr = PerDomainNormalizer().fit(dtr.train_series)
    nte = PerDomainNormalizer().fit(d.train_series)
    tr = ntr.transform(dtr.train_series)
    va = ntr.transform(dtr.val_series)
    te = nte.transform(d.test_series)
    W, S = int(cfg.data.window), int(cfg.data.stride)
    tr_w = sliding_windows(tr, W, S)
    va_w = sliding_windows(va, W, S)
    te_w = sliding_windows(te, W, S)
    tr_wl = window_label(dtr.train_labels, W, S)
    te_starts = list(range(0, len(te) - W + 1, S))

    mod = ContrastModule(
        n_ch=tr.shape[1],
        encoder=str(cfg.model.encoder),
        d=int(cfg.model.embedding_dim),
        tau=float(cfg.loss.tau),
        w_nce=float(cfg.loss.w_nce),
        w_ts2vec=float(cfg.loss.w_ts2vec),
        w_mask=float(cfg.loss.w_mask),
        lr=float(cfg.train.lr),
    )
    tr_l, va_l = make_loaders(
        tr_w, va_w, list(cfg.augment or ["jitter"]), int(cfg.train.batch_size), int(cfg.seed)
    )
    ckpt = Path("checkpoints")
    ckpt.mkdir(exist_ok=True)
    trainer = L.Trainer(
        max_epochs=int(cfg.train.epochs),
        accelerator="auto",
        enable_progress_bar=False,
        enable_model_summary=False,
        gradient_clip_val=1.0,
        precision="16-mixed" if __import__("torch").cuda.is_available() else 32,
        callbacks=[
            EarlyStopping("val_loss", patience=3, mode="min"),
            ModelCheckpoint(str(ckpt), monitor="val_loss", mode="min"),
        ],
    )
    trainer.fit(mod, tr_l, va_l)
    # prototype bank on NORMAL train windows only (train assumed normal)
    Ztr, _ = mod.embed(tr_w)
    clean = Ztr[tr_wl == 0]
    note = "clean-normal-windows"
    if len(clean) == 0:  # degenerate tiny splits: fall back to all windows (flagged, honest)
        clean, note = Ztr, "no-clean-windows-fallback"
        log.warning("no clean train windows; prototype bank fit on all train windows")
    bank = PrototypeBank(
        k=int(cfg.model.num_prototypes),
        method=str(cfg.get("scoring", {}).get("method", "kmeans")),
        seed=int(cfg.seed),
    ).fit(clean)
    Zte, _ = mod.embed(te_w)
    te_win_scores = bank.score(Zte)
    Zva, _ = mod.embed(va_w)
    va_win_scores = bank.score(Zva)
    va_ts = smooth(
        map_to_timestamps(va_win_scores, list(range(0, len(va) - W + 1, S)), len(va), W), 5
    )
    va_normal = dtr.val_labels == 0  # SOURCE val (no target labels in transfer)
    # Guard band: mapping/smoothing smears anomaly scores onto neighbours — exclude
    # timesteps within one window of any val anomaly so the reference stays clean.
    anom_idx = np.where(dtr.val_labels == 1)[0]
    if len(anom_idx):
        bad = np.zeros(len(va), bool)
        for a in anom_idx:
            bad[max(0, a - W) : a + W + 1] = True
        va_normal = va_normal & ~bad
    te_scores = smooth(
        map_to_timestamps(te_win_scores, te_starts, len(te), W),
        int(cfg.get("scoring", {}).get("smooth_width", 5)),
    )
    hw = float(cfg.get("scoring", {}).get("hybrid_weight", 0.7))
    # Hybrid with reconstruction error when the masked-recon aux trained the decoder (w_mask>0)
    if float(cfg.loss.w_mask) > 0:
        te_re = smooth(
            map_to_timestamps(mod.window_recon_error(te_w).mean(axis=1), te_starts, len(te), W), 5
        )
        va_re = smooth(
            map_to_timestamps(
                mod.window_recon_error(va_w).mean(axis=1),
                list(range(0, len(va) - W + 1, S)),
                len(va),
                W,
            ),
            5,
        )
        te_scores = hybrid(te_scores, te_re, hw)
        va_ts = hybrid(va_ts, va_re, hw)
        note += "+hybrid-recon"
    va_ref = va_ts[va_normal] if va_normal.sum() else va_ts
    va_ref_z, mu, sd = calibrate(va_ref, va_ref)  # val-only calibration (no test leakage)
    te_scores = (
        te_scores - mu
    ) / sd  # same val stats -> test (hybrid-aware: mu/sd fit post-hybrid)
    thr = select_threshold(
        str(cfg.threshold.method), va_ref_z, None, float(cfg.threshold.percentile)
    )
    rep = full_report(d.test_labels, te_scores, thr)
    log.info(f"contrastive@{domain_id}: strict_f1={rep['strict_f1']:.3f} pa_f1={rep['pa_f1']:.3f}")
    return {
        "mod": mod,
        "bank": bank,
        "W": W,
        "S": S,
        "d": d,
        "te": te,
        "te_w": te_w,
        "tel": d.test_labels,
        "starts": te_starts,
        "te_scores": te_scores,
        "thr": thr,
        "Zte": Zte,
        "note": note,
        "report": rep,
        "norm_mean": ntr.mean_,
        "norm_std": ntr.std_,
        "cal_mu": mu,
        "cal_sd": sd,
        "encoder_kind": str(cfg.model.encoder),
        "emb_d": int(cfg.model.embedding_dim),
        "smooth_width": int(cfg.get("scoring", {}).get("smooth_width", 5)),
    }


def run_single_domain(cfg, domain_id: str = "domain_0", test_domain_id: str | None = None) -> dict:
    """Thin wrapper: train + plot + results row (used by `train` entry point and E1/E2)."""
    art = prepare_and_train(cfg, domain_id, test_domain_id)
    tag = (
        domain_id
        if not test_domain_id or test_domain_id == domain_id
        else f"{domain_id}->{test_domain_id}"
    )
    plot_embeddings(
        art["Zte"],
        window_label(art["d"].test_labels, art["W"], art["S"]),
        f"reports/embed_{cfg.model.encoder}_{tag.replace('>', '')}.png",
        f"{cfg.model.encoder} {tag}",
    )
    row = {
        "model": f"contrast-{cfg.model.encoder}",
        "domain": tag,
        "proto_note": art["note"],
        **{k: round(float(v), 4) for k, v in art["report"].items() if isinstance(v, (int, float))},
    }
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
