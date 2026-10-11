"""Config system: OmegaConf+YAML with CLI overrides (dotlist key=value)."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def load_config(config_path: str | Path, overrides: list[str] | None = None):
    """Load YAML config and apply CLI dotlist overrides.

    Example: load_config("configs/default.yaml", ["model.encoder=transformer", "seed=1"])
    """
    from typing import Any

    cfg: Any = OmegaConf.load(str(config_path))
    if overrides:
        cli = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli)
    return cfg


def save_config(cfg, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, str(path))
