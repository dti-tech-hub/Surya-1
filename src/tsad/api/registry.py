"""Model registry: bundle = encoder weights + prototypes + norm/cal stats + config (one dir)."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import torch
from tsad.models.encoders import make_encoder
from tsad.models.prototype import PrototypeBank


def save_bundle(art: dict, cfg, path: str | Path) -> Path:
    """art: prepare_and_train() dict. Writes bundle dir {encoder.pt, prototypes.npy, stats.json, config.yaml}."""
    from tsad.config import save_config

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    mod = art["mod"]
    torch.save({"encoder": mod.enc.state_dict(), "kind": art["encoder_kind"],
                "emb_d": art["emb_d"]}, path / "encoder.pt")
    np.save(path / "prototypes.npy", art["bank"].prototypes)
    stats = {"norm_mean": np.asarray(art["norm_mean"], dtype=float).tolist(),
             "norm_std": np.asarray(art["norm_std"], dtype=float).tolist(),
             "cal_mu": float(art["cal_mu"]), "cal_sd": float(art["cal_sd"]),
             "threshold": float(art["thr"]), "W": int(art["W"]), "S": int(art["S"]),
             "smooth_width": int(art["smooth_width"]), "n_ch": int(np.asarray(art["norm_mean"]).shape[0]),
             "note": str(art["note"])}
    (path / "stats.json").write_text(json.dumps(stats, indent=1))
    save_config(cfg, path / "config.yaml")
    return path


class Bundle:
    """Loaded bundle: normalize -> window -> embed -> prototype score -> calibrate -> flag."""

    def __init__(self, path: str | Path):
        from omegaconf import OmegaConf

        path = Path(path)
        st = torch.load(path / "encoder.pt", map_location="cpu", weights_only=False)
        self.stats = json.loads((path / "stats.json").read_text())
        self.cfg = OmegaConf.load(path / "config.yaml")
        n_ch = self.stats["n_ch"]
        self.enc = make_encoder(st["kind"], n_ch, st["emb_d"])
        self.enc.load_state_dict(st["encoder"])
        self.enc.eval()
        self.bank = PrototypeBank(k=len(np.load(path / "prototypes.npy")))
        self.bank.proto_ = np.load(path / "prototypes.npy").astype(np.float64)
        self.mean_ = np.asarray(self.stats["norm_mean"])
        self.std_ = np.asarray(self.stats["norm_std"])

    def normalize(self, series: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(np.asarray(series, dtype=float))
        return (((x - self.mean_) / self.std_).astype(np.float32))

    @torch.no_grad()
    def detect(self, series: np.ndarray) -> dict:
        """series: (T, C) raw -> scores/flags/threshold (timestamp-level)."""
        from tsad.data.windows import sliding_windows
        from tsad.models.scoring import map_to_timestamps, smooth

        x = np.asarray(series, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.stats["n_ch"]:
            raise ValueError(f"expected (T, {self.stats['n_ch']}), got {x.shape}")
        W, S = self.stats["W"], self.stats["S"]
        if len(x) < W:
            raise ValueError(f"series length {len(x)} < window {W}")
        xn = self.normalize(x)
        w = sliding_windows(xn, W, S)
        reps = []
        for i in range(0, len(w), 256):
            reps.append(self.enc(torch.tensor(w[i:i + 256]))["rep"].numpy())
        Z = np.concatenate(reps)
        ts = smooth(map_to_timestamps(self.bank.score(Z), list(range(0, len(xn) - W + 1, S)),
                                      len(xn), W), self.stats["smooth_width"])
        z = (ts - self.stats["cal_mu"]) / self.stats["cal_sd"]
        thr = float(self.stats["threshold"])
        flags = (z >= thr).astype(int)
        return {"scores": [float(v) for v in z], "flags": [int(v) for v in flags],
                "threshold": thr, "n_flagged": int(flags.sum()), "T": len(x)}
