"""Encoders: TCN / Transformer (attn exposed) / ResNet1D. All: (B, T, C) -> rep (B, d)."""

from __future__ import annotations

import math

import torch
from torch import nn


class TCNBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, k: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, k, padding=(k - 1) * dilation, dilation=dilation)
        self.norm = nn.BatchNorm1d(ch)
        self.act = nn.ReLU()

    def forward(self, x):  # (B, C, T) causal: trim right padding
        y = self.conv(x)
        y = y[:, :, : x.shape[-1]]
        return self.act(self.norm(y) + x)


class TCNEncoder(nn.Module):
    def __init__(self, n_ch: int, d: int = 32, hidden: int = 32, levels: int = 3):
        super().__init__()
        self.inp = nn.Conv1d(n_ch, hidden, 1)
        self.blocks = nn.Sequential(*[TCNBlock(hidden, 2**i) for i in range(levels)])
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.mx = nn.AdaptiveMaxPool1d(1)
        # mean+max concat: mean captures shape, max preserves spike-type deviations
        self.fc = nn.Linear(2 * hidden, d)
        self.kind = "tcn"

    def forward(self, x):  # (B, T, C)
        h = self.inp(x.transpose(1, 2))  # (B, H, T)
        h = self.blocks(h)
        rep = self.fc(torch.cat([self.avg(h).squeeze(-1), self.mx(h).squeeze(-1)], dim=1))
        return {"rep": rep, "tokens": h.transpose(1, 2), "attn": None}


class PosEncoding(nn.Module):
    def __init__(self, d: int, max_len: int = 2048, learned: bool = False):
        super().__init__()
        if learned:
            self.pe = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
        else:
            pe = torch.zeros(max_len, d)
            pos = torch.arange(max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):  # (B, T, d)
        return x + self.pe[:, : x.shape[1]]


class AttnPool(nn.Module):
    """Attention pooling over time -> rep (B, d) + weights (B, T)."""

    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) * 0.02)

    def forward(self, h):  # (B, T, d)
        w = torch.softmax((h @ self.q), dim=1)  # (B, T)
        return (h * w.unsqueeze(-1)).sum(1), w


class TransformerEncoder(nn.Module):
    def __init__(self, n_ch: int, d: int = 32, hidden: int = 32, heads: int = 2, layers: int = 2):
        super().__init__()
        self.inp = nn.Linear(n_ch, hidden)
        self.pos = PosEncoding(hidden)
        layer = nn.TransformerEncoderLayer(
            hidden, heads, dim_feedforward=4 * hidden, batch_first=True
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=layers)
        self.pool = AttnPool(hidden)
        self.fc = nn.Linear(hidden, d)
        self.kind = "transformer"

    def forward(self, x):  # (B, T, C)
        h = self.tr(self.pos(self.inp(x)))  # (B, T, H)
        rep, w = self.pool(h)
        return {"rep": self.fc(rep), "tokens": h, "attn": w}


class ResBlock1D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 5, padding=2)
        self.c2 = nn.Conv1d(ch, ch, 5, padding=2)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.c2(self.act(self.c1(x))))


class ResNet1DEncoder(nn.Module):
    def __init__(self, n_ch: int, d: int = 32, hidden: int = 32, blocks: int = 2):
        super().__init__()
        self.inp = nn.Conv1d(n_ch, hidden, 7, padding=3)
        self.body = nn.Sequential(*[ResBlock1D(hidden) for _ in range(blocks)])
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.mx = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(2 * hidden, d)
        self.kind = "resnet"

    def forward(self, x):  # (B, T, C)
        h = self.body(self.inp(x.transpose(1, 2)))
        rep = self.fc(torch.cat([self.avg(h).squeeze(-1), self.mx(h).squeeze(-1)], dim=1))
        return {"rep": rep, "tokens": h.transpose(1, 2), "attn": None}


def make_encoder(kind: str, n_ch: int, d: int = 32, **kw) -> nn.Module:
    kinds = {"tcn": TCNEncoder, "transformer": TransformerEncoder, "resnet": ResNet1DEncoder}
    if kind not in kinds:
        raise ValueError(f"unknown encoder {kind}; choose {sorted(kinds)}")
    return kinds[kind](n_ch, d, **kw)
