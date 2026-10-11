"""Phase-0 tests: config overrides, seed determinism, smoke pipeline."""
import numpy as np
from tsad.config import load_config
from tsad.seed import set_seed


def test_config_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("seed: 1\nmodel:\n  encoder: tcn\n")
    cfg = load_config(str(p), ["seed=7", "model.encoder=transformer"])
    assert int(cfg.seed) == 7
    assert str(cfg.model.encoder) == "transformer"


def test_seed_deterministic():
    set_seed(0)
    a = np.random.rand(5)
    set_seed(0)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_smoke_pipeline():
    cfg = load_config("configs/smoke.yaml")
    assert str(cfg.tracking.backend) == "off"
    assert int(cfg.data.n_domains) == 2
