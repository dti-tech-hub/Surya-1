"""Phase-10 tests: report builds from fixture CSVs, NOT RUN placeholders, no self-ingestion."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from scripts.make_report import build, draft_results


def _csv(p: Path, rows: list[dict]):
    import csv

    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_build_from_real_numbers(tmp_path):
    rep = tmp_path / "reports"
    rep.mkdir()
    _csv(
        rep / "adapt_smoke.csv",
        [
            {"condition": "source-only", "strict_f1": "0.10", "auc_roc": "0.8"},
            {"condition": "adaptive-full", "strict_f1": "0.40", "auc_roc": "0.9"},
        ],
    )
    res = build(rep, tmp_path / "final")
    assert "adapt_smoke" in res["tables"]
    txt = (tmp_path / "final" / "results_section.md").read_text()
    assert "0.400" in txt and "adaptive-full" in txt  # real numbers surface
    assert "NOT RUN" in txt  # missing sections flagged, not fabricated


def test_empty_reports_all_not_run(tmp_path):
    rep = tmp_path / "reports"
    rep.mkdir()
    res = build(rep, tmp_path / "final")
    assert res["tables"] == [] and res["figures"] == []
    txt = (tmp_path / "final" / "results_section.md").read_text()
    assert txt.count("NOT RUN") >= 2


def test_no_self_ingestion(tmp_path):
    rep = tmp_path / "reports"
    rep.mkdir()
    _csv(rep / "e1_smoke.csv", [{"model": "a", "strict_f1": "0.5"}])
    build(rep, tmp_path / "final")
    n1 = (tmp_path / "final" / "results_section.md").read_text()
    build(rep, tmp_path / "final")  # rerun must not ingest final/ outputs
    n2 = (tmp_path / "final" / "results_section.md").read_text()
    assert n1 == n2
    assert "draft_results" not in n2  # no leakage of generator internals
    assert draft_results({}).count("NOT RUN") >= 1
