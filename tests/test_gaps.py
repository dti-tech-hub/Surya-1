"""Gap-closing tests for uncovered-but-real paths (plots, pseudo-anomalies, download utils)."""

import numpy as np

from tsad.data.base import DatasetMissingError
from tsad.data.download import require, sha256, verify_checksum
from tsad.data.pseudo_anomaly import inject_pseudo
from tsad.data.windows import window_mask
from tsad.eval.bootstrap import aggregate_seeds
from tsad.eval.metrics import event_wise_f1, range_based_f1
from tsad.eval.threshold import select_threshold
from tsad.explain.attention_exp import attention_heatmap, attention_part
from tsad.explain.prototype_exp import overlay_plot


def test_pseudo_anomaly():
    x = np.zeros((20, 2), dtype=np.float32)
    y, m = inject_pseudo(x, "spike", seed=0)
    assert y.shape == x.shape and m.sum() >= 1
    y2, m2 = inject_pseudo(x + 1.0, "scale", seed=1)
    assert m2.sum() >= 1 and not np.array_equal(y2, x + 1.0)
    try:
        inject_pseudo(x, "nope")
        raise AssertionError("should raise")
    except ValueError:
        pass


def test_download_helpers(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    assert require(p, "x") == p
    assert verify_checksum(p, sha256(p)) is True
    assert verify_checksum(p, "deadbeef") is False
    assert verify_checksum(p, None) is True
    try:
        require(tmp_path / "missing", "get it at URL")
        raise AssertionError("should raise")
    except DatasetMissingError as e:
        assert "URL" in str(e)


def test_window_mask():
    m = np.zeros((10, 2), dtype=int)
    m[5, 1] = 1
    w = window_mask(m, window=4, stride=3)
    assert w.shape == (3, 4, 2) and w[:, :, 1].sum() >= 1


def test_plots_and_attention(tmp_path):
    x = np.random.default_rng(0).normal(size=(16, 2)).astype(float)
    overlay_plot(x, x * 0.5, tmp_path / "ov.png")
    attention_heatmap(np.ones(16) / 16, tmp_path / "ah.png")
    a = attention_part(np.ones(8) / 8)
    assert a["available"] and abs(sum(a["weights"]) - 1.0) < 1e-9
    assert attention_part(None)["available"] is False
    assert (tmp_path / "ov.png").exists() and (tmp_path / "ah.png").exists()


def test_threshold_branches():
    s = np.random.default_rng(0).random(200)
    assert np.isfinite(select_threshold("best-f1", s, np.zeros(200, dtype=int)))
    assert select_threshold("fixed", s, fixed=0.3) == 0.3
    try:
        select_threshold("nope", s)
        raise AssertionError("should raise")
    except ValueError:
        pass


def test_event_range_edges_and_agg():
    y = np.zeros(10, dtype=int)
    s = np.zeros(10)
    assert event_wise_f1(y, s, 0.5)["event_f1"] == 1.0  # no events either side
    assert range_based_f1(y, s, 0.5)["range_f1"] == 1.0
    agg = aggregate_seeds([{"f1": 0.5}])
    assert agg["f1_std"] == 0.0 and agg["n_seeds"] == 1
