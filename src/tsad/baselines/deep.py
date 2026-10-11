"""Deep reconstruction baselines (PyTorch): LSTM-AE, Conv-AE, Transformer-AE.

Score = window reconstruction error mapped to timestamps (mean over covering windows).
CPU-friendly defaults (hidden=16, epochs configurable; smoke uses epochs=1).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from tsad.baselines.base import Baseline


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMAE(nn.Module):
    def __init__(self, n_ch: int, hidden: int = 16):
        super().__init__()
        self.enc = nn.LSTM(n_ch, hidden, batch_first=True)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_ch)

    def forward(self, x):  # x: (B, T, C)
        z, _ = self.enc(x)
        d, _ = self.dec(z)
        return self.out(d)


class ConvAE(nn.Module):
    def __init__(self, n_ch: int, hidden: int = 16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(n_ch, hidden, 7, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.ReLU(),
        )
        self.dec = nn.Conv1d(hidden, n_ch, 5, padding=2)

    def forward(self, x):  # (B, T, C) -> transpose for Conv1d
        h = self.enc(x.transpose(1, 2))
        return self.dec(h).transpose(1, 2)


class TransAE(nn.Module):
    def __init__(self, n_ch: int, hidden: int = 16, heads: int = 2):
        super().__init__()
        self.inp = nn.Linear(n_ch, hidden)
        layer = nn.TransformerEncoderLayer(
            hidden, heads, dim_feedforward=4 * hidden, batch_first=True
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=1)
        self.out = nn.Linear(hidden, n_ch)

    def forward(self, x):  # (B, T, C)
        return self.out(self.tr(self.inp(x)))


class AEModel(Baseline):
    """Generic windowed AE trainer shared by the three architectures."""

    def __init__(
        self,
        kind: str = "lstm",
        hidden: int = 16,
        window: int = 32,
        epochs: int = 3,
        lr: float = 1e-3,
        batch: int = 32,
        seed: int = 0,
    ):
        assert kind in ("lstm", "conv", "trans")
        self.kind, self.hidden, self.window = kind, hidden, window
        self.epochs, self.lr, self.batch, self.seed = epochs, lr, batch, seed
        self.name = f"{kind}_ae"

    def _net(self, c: int) -> nn.Module:
        return {"lstm": LSTMAE, "conv": ConvAE, "trans": TransAE}[self.kind](c, self.hidden)

    def fit(self, train: np.ndarray) -> AEModel:
        torch.manual_seed(self.seed)
        dev = _device()
        net = self._net(train.shape[1]).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        W = min(self.window, len(train))
        # window the train series (stride W//2), normalize-ish scale already applied upstream
        idx = list(range(0, len(train) - W + 1, max(1, W // 2))) or [0]
        xb = torch.tensor(np.stack([train[i : i + W] for i in idx]), dtype=torch.float32)
        net.train()
        for _ in range(max(1, self.epochs)):
            perm = torch.randperm(len(xb))
            for b in range(0, len(xb), self.batch):
                opt.zero_grad()
                bb = xb[perm[b : b + self.batch]].to(dev)
                loss = loss_fn(net(bb), bb)
                loss.backward()
                opt.step()
        self.net_, self.W_ = net.eval(), W
        return self

    @torch.no_grad()
    def score(self, series: np.ndarray) -> np.ndarray:
        dev = _device()
        T = len(series)
        W = min(self.W_, T)
        idx = list(range(0, T - W + 1, max(1, W // 2))) or [0]
        errs = np.zeros(T)
        counts = np.zeros(T)
        for i in idx:
            w = torch.tensor(series[i : i + W][None], dtype=torch.float32).to(dev)
            e = ((self.net_(w)[0].cpu().numpy() - series[i : i + W]) ** 2).mean(axis=1)
            errs[i : i + W] += e
            counts[i : i + W] += 1
        return (errs / np.maximum(1, counts)).astype(np.float64)
