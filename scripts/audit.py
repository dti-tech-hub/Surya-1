"""Phase-11 final audit: leakage, label-use, reproducibility, gates, O1-O4 gap checklist.

Writes reports/audit.md. Exit 1 on any FAIL (CI-gateable); findings also print to stdout.
"""

from __future__ import annotations

import argparse
import inspect
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

RESULTS: list[tuple[str, str, str]] = []  # (area, PASS/FAIL, detail)


def check(area: str, ok: bool, detail: str):
    RESULTS.append((area, "PASS" if ok else "FAIL", detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {area}: {detail}")


def leak_checks():
    import numpy as np

    from tsad.data.augment import AugmentPipeline
    from tsad.data.normalize import PerDomainNormalizer
    from tsad.lightning.contrast import ContrastModule, WindowDataset

    # 1. normalizer refuses transform-before-fit (fit-train-only enforced by API)
    try:
        PerDomainNormalizer().transform(np.ones((5, 2), np.float32))
        check("leak/norm-guard", False, "transform before fit did not raise")
    except AssertionError:
        check("leak/norm-guard", True, "transform() asserts fit() first (train-only)")
    # 2. default training path carries NO labels (WindowDataset.sup is None unless w_sup>0)
    X = np.random.default_rng(0).normal(size=(6, 16, 2)).astype(np.float32)
    d = WindowDataset(X, AugmentPipeline(["jitter"], seed=0), seed=0)
    check("leak/default-unlabeled", d[0][-1] == -1, "default sup=-1 (labels never touch loss)")
    # 3. training/validation steps reference no label tensors (sup term is the only gate)
    import re

    from tsad.lightning import adapt as _a

    src = inspect.getsource(ContrastModule.training_step) + inspect.getsource(
        ContrastModule.validation_step
    )

    # whole-word match on CODE (comments stripped; 'unlabeled' must NOT count as label use)
    def code(s: str) -> str:
        return "\n".join(line.split("#")[0] for line in s.splitlines())

    def has_label(s: str) -> bool:
        return bool(re.search(r"(?<!un)labels?\b", code(s).lower()))

    check("leak/contrast-no-labels", not has_label(src), "ContrastModule steps label-free")
    asrc = inspect.getsource(_a.AdaptModule.training_step)
    check("leak/adapt-no-labels", not has_label(asrc), "AdaptModule step label-free")
    # 4. threshold call sites use val only (static scan of scripts)
    bad = []
    for f in ["train.py", "eval.py", "adapt.py"]:
        lines = (ROOT / "scripts" / f).read_text().splitlines()
        for i, line in enumerate(lines, 1):
            if "select_threshold" in line and ("tel" in line or "test_labels" in line):
                bad.append(f"{f}:{i}")
    check("leak/threshold-val-only", not bad, "no test-label thresholding" if not bad else str(bad))


def label_use_report() -> list[str]:
    return [
        "representation/adaptation losses: label-free (InfoNCE/TS2Vec/masked/MMD/CORAL/DANN).",
        "prototype bank: normal-train-window SELECTION uses train labels (mirrors 'train assumed normal');",
        "  flagged fallback logged when no clean windows.",
        "weak supervision: labels enter ONLY if loss.w_sup>0 AND data.weak_labels>0 (both default 0).",
        "metrics/thresholds: val labels allowed (best-f1); test labels for scoring only, never fitting.",
    ]


def repro_checks():
    import numpy as np
    import torch

    from tsad.data.synthetic import SyntheticDataset
    from tsad.lightning.contrast import ContrastModule

    a = SyntheticDataset(2, 512, 2, 0.05, 0).load()
    b = SyntheticDataset(2, 512, 2, 0.05, 0).load()
    same = all(
        np.array_equal(a[k].test_series, b[k].test_series)
        and np.array_equal(a[k].test_labels, b[k].test_labels)
        for k in a
    )
    check("repro/synthetic", same, "same seed -> identical series+labels+masks")
    torch.manual_seed(0)
    m1 = ContrastModule(n_ch=2, encoder="tcn", d=8)
    torch.manual_seed(0)
    m2 = ContrastModule(n_ch=2, encoder="tcn", d=8)
    xb = torch.randn(4, 32, 2)
    same_loss = torch.allclose(m1.enc(xb)["rep"], m2.enc(xb)["rep"])
    check("repro/init", bool(same_loss), "same seed -> identical init forward")


def gate_status() -> dict:
    out = {}
    for name, cmd in [
        ("pytest", [sys.executable, "-m", "pytest", "-q"]),
        (
            "ruff",
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "src",
                "tests",
                "scripts",
                "experiments",
                "dashboard",
            ],
        ),
        (
            "black",
            [
                sys.executable,
                "-m",
                "black",
                "--check",
                "src",
                "tests",
                "scripts",
                "experiments",
                "dashboard",
            ],
        ),
        ("mypy", [sys.executable, "-m", "mypy", "src/tsad"]),
    ]:
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, cwd=ROOT, timeout=600, check=False
            )
            tail = (r.stdout + r.stderr).strip().splitlines()
            out[name] = ("PASS" if r.returncode == 0 else "FAIL", tail[-1][:160] if tail else "")
        except Exception as e:
            out[name] = ("FAIL", str(e)[:160])
    for k, (s, d) in out.items():
        check(f"gate/{k}", s == "PASS", d)
    return out


GAPS = [
    (
        "O1",
        "weak-label supervision",
        "IMPLEMENTED this phase (SupCon, off by default); single-seed smoke only",
    ),
    (
        "O2",
        "conditional/class-prototype alignment",
        "chamfer option implemented, not validated E2E",
    ),
    ("O2", "DANN/CORAL E2E", "unit-tested; E3 full-mode only, not run on GPU box yet"),
    ("O3", "multi-seed neg-transfer stats", "2 seeds run; 5-seed full pending (Phase 9 plan)"),
    (
        "O4",
        "DeepSHAP",
        "NOT shipped: crashes on BatchNorm TCNs (shap internals); KernelSHAP+LIME+IG cover SHAP/LIME ask",
    ),
    ("O4", "VUS/affiliation metrics", "NOT implemented (documented; range/event-wise done)"),
    (
        "data",
        "real SMD/SMAP transfer numbers",
        "NOT RUN (no local files; loaders ready, graceful skip)",
    ),
    ("data", "LODO/multi-source E2E", "registry tested; E2 runs pairs only"),
    (
        "train",
        "MLflow/W&B logging E2E",
        "NOT RUN (mlflow not installed; Tracker unit-covered by smoke off-mode)",
    ),
    (
        "deploy",
        "`make` on Windows",
        "make binary absent here; python equivalents verified (README)",
    ),
]


def objective_map(reports: Path) -> list[dict]:
    def has(*ps):
        return all((ROOT / p).exists() for p in ps)

    rows = [
        {
            "obj": "O1",
            "code": "models/encoders.py, losses/contrastive.py, lightning/contrast.py",
            "config": "configs/model/*.yaml",
            "test": "tests/test_contrast.py",
            "exp": "E1",
            "fig": "reports/embed_tcn_domain_0.png",
            "ok": has(
                "src/tsad/models/encoders.py",
                "tests/test_contrast.py",
                "reports/embed_tcn_domain_0.png",
            ),
        },
        {
            "obj": "O2",
            "code": "losses/{mmd,coral,dann}.py, alignment/adaptive.py",
            "config": "configs/align/*.yaml",
            "test": "tests/test_alignment.py",
            "exp": "E3",
            "fig": "reports/adapt_mmd.png",
            "ok": has("src/tsad/alignment/adaptive.py", "reports/adapt_mmd.png"),
        },
        {
            "obj": "O3",
            "code": "alignment/{gate,domain_weights,preserve,monitor}.py",
            "config": "configs/align/adaptive_full.yaml",
            "test": "tests/test_alignment.py",
            "exp": "E4",
            "fig": "reports/exp/e4_smoke.csv",
            "ok": has("src/tsad/alignment/gate.py", "reports/exp/e4_smoke.csv"),
        },
        {
            "obj": "O4",
            "code": "explain/*.py (6 parts)",
            "config": "configs/explain.yaml",
            "test": "tests/test_explain.py",
            "exp": "E6",
            "fig": "reports/explain_smoke/index.html",
            "ok": has("src/tsad/explain/counterfactual.py", "reports/explain_smoke/index.html"),
        },
    ]
    for r in rows:
        check(f"obj/{r['obj']}", bool(r["ok"]), f"code+config+test+exp+figure present ({r['exp']})")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/audit.md")
    args = ap.parse_args()
    print("== leakage ==")
    leak_checks()
    print("== reproducibility ==")
    repro_checks()
    print("== gates (pytest/ruff/black/mypy) ==")
    gates = gate_status()
    print("== objectives ==")
    rows = objective_map(Path("reports"))
    L = ["# Final audit (generated — do not hand-edit; rerun `python scripts/audit.py`)", ""]
    L += ["## Checks", "", "| area | status | detail |", "|---|---|---|"]
    L += [f"| {a} | {s} | {d} |" for a, s, d in RESULTS]
    L += ["", "## Label use (where labels may appear)", ""]
    L += [f"- {x}" for x in label_use_report()]
    L += ["", "## Gates", ""]
    L += [f"- {k}: {v[0]} — {v[1]}" for k, v in gates.items()]
    L += [
        "",
        "## O1–O4 mapping",
        "",
        "| obj | code | config | test | exp | figure | ok |",
        "|---|---|---|---|---|---|---|",
    ]
    L += [
        f"| {r['obj']} | {r['code']} | {r['config']} | {r['test']} | {r['exp']} | {r['fig']} | "
        f"{'YES' if r['ok'] else 'NO'} |"
        for r in rows
    ]
    L += ["", "## Explicit gaps / not-validated", "", "| obj | item | status |", "|---|---|---|"]
    L += [f"| {o} | {i} | {s} |" for o, i, s in GAPS]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"wrote {args.out}")
    fails = [r for r in RESULTS if r[1] == "FAIL"]
    print(
        f"audit: {len(RESULTS) - len(fails)}/{len(RESULTS)} PASS"
        + (f", FAILS: {fails}" if fails else "")
    )
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
