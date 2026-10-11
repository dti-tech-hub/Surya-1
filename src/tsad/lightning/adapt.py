"""Adaptation Lightning module: contrastive on sources + adaptive alignment to UNLABELED target.

Labels unused everywhere in training (self-supervised + unsupervised alignment).
"""

from __future__ import annotations

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tsad.alignment.adaptive import AdaptiveAligner
from tsad.alignment.monitor import NegTransferMonitor
from tsad.alignment.preserve import compactness
from tsad.data.augment import AugmentPipeline
from tsad.losses.contrastive import info_nce
from tsad.losses.dann import dann_lambda
from tsad.models.encoders import make_encoder


class PairedDataset(Dataset):
    """Item: (src_view1, src_view2, src_domain_idx, tgt_window). Target windows raw (unlabeled)."""

    def __init__(
        self,
        src_windows: list[np.ndarray],
        tgt_windows: np.ndarray,
        pipe: AugmentPipeline,
        seed: int = 0,
    ):
        self.src = [s.astype(np.float32) for s in src_windows]
        self.tgt = tgt_windows.astype(np.float32)
        self.pipe = pipe
        self.seed = seed
        self.n = max(len(tgt_windows), 1)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed + i)
        k = int(rng.integers(0, len(self.src)))
        x = self.src[k][int(rng.integers(0, len(self.src[k])))]
        v1, v2 = self.pipe.two_views(x)
        t = self.tgt[int(rng.integers(0, len(self.tgt)))]
        return torch.from_numpy(v1), torch.from_numpy(v2), k, torch.from_numpy(t)


class AdaptModule(L.LightningModule):
    def __init__(
        self,
        n_ch: int,
        n_src: int,
        encoder: str = "tcn",
        d: int = 32,
        tau: float = 0.1,
        lr: float = 1e-3,
        align_cfg: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        ac = align_cfg or {}
        self.enc = make_encoder(encoder, n_ch, d)
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(d, d), torch.nn.ReLU(), torch.nn.Linear(d, 32)
        )
        self.aligner = AdaptiveAligner(
            d,
            method=ac.get("method", "mmd"),
            use_gate=ac.get("use_gate", True),
            use_weights=ac.get("use_weights", True),
            alpha=ac.get("alpha", 0.5),
            beta=ac.get("beta", 0.1),
            gamma=ac.get("gamma", 0.01),
            warmup_epochs=ac.get("warmup", 1),
            cond=ac.get("cond", False),
            tau=ac.get("tau", 1.0),
        )
        self.monitor = NegTransferMonitor() if ac.get("use_monitor", True) else None
        self.n_src = n_src
        self._last_weights = None
        self._dom_cache: tuple[list[torch.Tensor], torch.Tensor] | None = None
        self._epoch_w: torch.Tensor | None = None

    def set_domain_cache(self, src_full: list[np.ndarray], tgt_full: np.ndarray) -> None:
        """Full train windows (input space) for stable epoch-level domain weights (detached)."""
        self._dom_cache = (
            [torch.tensor(s.reshape(len(s), -1), dtype=torch.float32) for s in src_full],
            torch.tensor(tgt_full.reshape(len(tgt_full), -1), dtype=torch.float32),
        )

    def on_train_epoch_start(self) -> None:
        if self._dom_cache is None or self.aligner.weighter is None:
            return
        srcs, tgt = self._dom_cache
        dev = next(self.parameters()).device
        with torch.no_grad():
            self._epoch_w = self.aligner.weighter([s.to(dev) for s in srcs], tgt.to(dev)).detach()

    def _reps(self, x):
        return self.enc(x)["rep"]

    def training_step(self, batch, _):
        v1, v2, k, t = batch
        o1, o2 = self.enc(v1), self.enc(v2)
        lc = info_nce(self.proj(o1["rep"]), self.proj(o2["rep"]), self.hparams.tau)
        # per-source NORMAL reps: use all (labels unused); gate over gated space
        zt = self._reps(t)
        srcs = [self._reps(v1[k == j]) for j in range(self.n_src) if (k == j).sum() > 1]
        out = self.aligner(
            srcs,
            zt,
            self.current_epoch,
            self.trainer.max_epochs,
            dann_lambda(self.current_epoch / max(1, self.trainer.max_epochs)),
            fixed_weights=self._epoch_w,
        )
        loss = lc + out["total_align"]
        self.log_dict(
            {"train_loss": loss, "train_contrast": lc, "train_align": out["align"]}, prog_bar=True
        )
        self._last_weights = out["weights"]
        return loss

    def validation_step(self, batch, _):
        v1, v2, _k, t = batch
        o1, o2 = self.enc(v1), self.enc(v2)
        self.log("val_contrast", info_nce(self.proj(o1["rep"]), self.proj(o2["rep"])))
        self.log("val_compact", compactness(self._reps(t)))

    def on_validation_epoch_end(self):
        if self.monitor is None:
            return
        vc = float(self.trainer.callback_metrics.get("val_contrast", 0))
        cp = float(self.trainer.callback_metrics.get("val_compact", 0))
        info = self.monitor.update(self, vc, cp)
        self.aligner.alpha_mult = info["alpha_mult"]
        self.log_dict({"mon_proxy": info["proxy"], "mon_alpha": info["alpha_mult"]})

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return [opt], [torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)]

    @torch.no_grad()
    def embed(self, X: np.ndarray, batch: int = 256) -> np.ndarray:
        self.eval()
        dev = next(self.parameters()).device
        outs = []
        for i in range(0, len(X), batch):
            outs.append(
                self._reps(torch.from_numpy(X[i : i + batch].astype(np.float32)).to(dev))
                .cpu()
                .numpy()
            )
        return np.concatenate(outs)


def make_paired_loaders(
    src_w: list[np.ndarray],
    tgt_w: np.ndarray,
    src_val: list[np.ndarray],
    tgt_val: np.ndarray,
    augments: list[str],
    batch: int,
    seed: int = 0,
):
    pipe = AugmentPipeline(augments or ["jitter"], seed=seed)
    tr = DataLoader(
        PairedDataset(src_w, tgt_w, pipe, seed), batch_size=batch, shuffle=True, num_workers=0
    )
    va = DataLoader(
        PairedDataset(src_val, tgt_val, pipe, seed + 1), batch_size=batch, num_workers=0
    )
    return tr, va
