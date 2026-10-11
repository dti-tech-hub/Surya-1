"""Phase-11 tests: audit helpers on fixtures (fast; full gates run inside audit.py)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def test_label_use_report_documents_weak_gating():
    from scripts.audit import label_use_report

    txt = "\n".join(label_use_report())
    assert "w_sup" in txt and "default 0" in txt


def test_objective_map_structure(tmp_path):
    from scripts.audit import objective_map

    rows = objective_map(tmp_path)
    assert {r["obj"] for r in rows} == {"O1", "O2", "O3", "O4"}
    assert all(set(r) >= {"code", "config", "test", "exp", "fig", "ok"} for r in rows)


def test_gaps_list_explicit():
    from scripts.audit import GAPS

    text = "\n".join(f"{o} {i} {s}" for o, i, s in GAPS)
    assert "DeepSHAP" in text and "VUS" in text
    assert any("NOT" in s for _, _, s in GAPS)
