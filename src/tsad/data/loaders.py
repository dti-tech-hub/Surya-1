"""Dataset factory: name -> domains. Real loaders raise DatasetMissingError when files absent."""

from __future__ import annotations

from tsad.data.base import BaseTSADDataset


def get_dataset(name: str, cfg) -> BaseTSADDataset:
    """cfg: OmegaConf/dotdict with data.* fields. Synthetic honors weak-label-agnostic params."""
    from tsad.data.datasets import SMAPMSLDataset, SMDDataset, SWaTDataset, UCRDataset, YahooDataset
    from tsad.data.synthetic import SyntheticDataset

    d = cfg.data
    if name == "synthetic":
        return SyntheticDataset(
            int(d.n_domains),
            int(d.length),
            int(d.n_channels),
            float(d.anomaly_ratio),
            int(cfg.seed),
        )
    if name == "smd":
        return SMDDataset(root=str(getattr(d, "root", "data/smd")))
    if name in ("smap", "msl", "smap_msl"):
        return SMAPMSLDataset(root=str(getattr(d, "root", "data/smap_msl")))
    if name == "ucr":
        return UCRDataset(root=str(getattr(d, "root", "data/ucr")))
    if name == "swat":
        return SWaTDataset()
    if name == "yahoo":
        return YahooDataset()
    raise ValueError(f"unknown dataset {name!r}; choose synthetic|smd|smap_msl|ucr|swat|yahoo")
