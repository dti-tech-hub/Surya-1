"""Phase-6 tests: table helpers, stats, aggregation, fast experiment runs."""
import numpy as np
from experiments.common import to_latex, paired_stats, mean_std_table, save_all


def test_latex_and_save(tmp_path):
    rows = [{"model": "a", "strict_f1": 0.5}, {"model": "b", "strict_f1": 0.7}]
    tex = to_latex(rows)
    assert "\\begin{table}" in tex and "0.500" in tex
    outs = save_all(rows, str(tmp_path / "t"))
    import os

    assert all(os.path.exists(p) for p in outs.values())


def test_paired_stats_needs_seeds():
    assert "n/a" in paired_stats([0.5], [0.6])["t_p"]
    r = paired_stats([0.5, 0.6, 0.7], [0.4, 0.5, 0.6])
    assert 0.0 <= r["t_p"] <= 1.0 and 0.0 <= r["w_p"] <= 1.0


def test_mean_std_table():
    rows = [{"model": "a", "strict_f1": 0.5}, {"model": "a", "strict_f1": 0.7}]
    agg = mean_std_table(rows, ("model",))
    assert agg[0]["strict_f1_mean"] == 0.6 and agg[0]["n_seeds"] == 2


def test_e8_fast_and_dispatcher(tmp_path):
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]  # repo root (no hardcoded paths)
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    from tsad.config import load_config
    from experiments import e8_efficiency

    cfg = load_config("configs/smoke.yaml")
    rows = e8_efficiency.run(cfg, [0], str(tmp_path), smoke=True)
    assert len(rows) == 2 and all(r["params"] > 0 for r in rows)
    assert all(r["score_windows_per_sec"] > 0 for r in rows)
