"""Lightning contrastive module + window data module (self-supervised, labels unused in training).

L = w_nce * InfoNCE + w_ts2vec * hierarchical + w_mask * masked-recon.
Validation monitors contrastive loss on val windows (unlabeled proxy — no labels).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightning as L
from tsad.models.encoders import make_encoder
from tsad.losses.contrastive import info_nce, ts2vec_loss, masked_mse
from tsad.data.augment import AugmentPipeline


class WindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, pipe: AugmentPipeline, mask_ratio: float = 0.15,
                 seed: int = 0):
        self.X = windows.astype(np.float32)
        self.pipe = pipe
        self.mask_ratio = mask_ratio
        self.seed = seed

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed + i)
        x = self.X[i]
        v1, v2 = self.pipe.two_views(x)
        T = x.shape[0]
        m = (rng.random(T) < self.mask_ratio).astype(np.float32)
        return torch.from_numpy(v1), torch.from_numpy(v2), torch.from_numpy(x), torch.from_numpy(m)


class ContrastModule(L.LightningModule):
    def __init__(self, n_ch: int, encoder: str = "tcn", d: int = 32, p: int = 32,
                 tau: float = 0.1, w_nce: float = 1.0, w_ts2vec: float = 0.0,
                 w_mask: float = 0.0, lr: float = 1e-3, hid: int = 32):
        super().__init__()
        self.save_hyperparameters()
        self.enc = make_encoder(encoder, n_ch, d, hidden=hid)
        self.proj = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, p))
        self.dec = nn.Linear(hid, n_ch)  # per-step recon from tokens (masked aux)

    def _loss(self, v1, v2, x, m):
        o1, o2 = self.enc(v1), self.enc(v2)
        z1, z2 = self.proj(o1["rep"]), self.proj(o2["rep"])
        loss = 0.0
        if self.hparams.w_nce > 0:
            loss = loss + self.hparams.w_nce * info_nce(z1, z2, self.hparams.tau)
        if self.hparams.w_ts2vec > 0:
            loss = loss + self.hparams.w_ts2vec * ts2vec_loss(o1["tokens"], o2["tokens"])
        if self.hparams.w_mask > 0:
            # decode masked view1 tokens back to clean x
            pr = self.dec(o1["tokens"])
            loss = loss + self.hparams.w_mask * masked_mse(pr, x, m)
        return loss

    def training_step(self, batch, _):
        v1, v2, x, m = batch
        loss = self._loss(v1, v2, x, m)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        v1, v2, x, m = batch
        self.log("val_loss", self._loss(v1, v2, x, m), prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

    @torch.no_grad()
    def window_recon_error(self, X: np.ndarray, batch: int = 256) -> np.ndarray:
        """Per-timestep MSE of raw windows through enc->dec. X: (N, W, C) -> (N, W)."""
        self.eval()
        dev = next(self.parameters()).device
        errs = []
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch].astype(np.float32)).to(dev)
            pr = self.dec(self.enc(xb)["tokens"])
            errs.append(((pr - xb) ** 2).mean(-1).cpu().numpy())
        return np.concatenate(errs).astype(np.float64)

    @torch.no_grad()
    def embed(self, X: np.ndarray, batch: int = 256) -> tuple[np.ndarray, np.ndarray | None]:
        self.eval()
        dev = next(self.parameters()).device
        reps, attns = [], []
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch].astype(np.float32)).to(dev)
            o = self.enc(xb)
            reps.append(o["rep"].cpu().numpy())
            attns.append(None if o["attn"] is None else o["attn"].cpu().numpy())
        Z = np.concatenate(reps)
        A = np.concatenate(attns) if attns[0] is not None else None
        return Z, A


def make_loaders(tr_w: np.ndarray, va_w: np.ndarray, augments: list[str], batch: int,
                 seed: int = 0) -> tuple[DataLoader, DataLoader]:
    pipe = AugmentPipeline(augments or ["jitter"], seed=seed)
    return (DataLoader(WindowDataset(tr_w, pipe, seed=seed), batch_size=batch, shuffle=True,
                       num_workers=0),
            DataLoader(WindowDataset(va_w, pipe, seed=seed + 1), batch_size=batch,
                       num_workers=0))
