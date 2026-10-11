"""Experiment dispatcher: --exp e1..e8 | --all-smoke | --all. Multi-seed, tables+plots+stats."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from tsad.config import load_config
from tsad.logging_utils import get_logger
from experiments import common as C
from experiments import (e1_singledomain, e2_transfer, e3_alignment, e4_negtransfer,
                         e5_ablation, e6_explain, e7_sensitivity, e8_efficiency)

EXPS = {"e1": e1_singledomain, "e2": e2_transfer, "e3": e3_alignment, "e4": e4_negtransfer,
        "e5": e5_ablation, "e6": e6_explain, "e7": e7_sensitivity, "e8": e8_efficiency}
TITLES = {"e1": "E1 single-domain", "e2": "E2 transfer pairs", "e3": "E3 alignment comparison",
          "e4": "E4 negative transfer", "e5": "E5 ablations", "e6": "E6 explainability",
          "e7": "E7 sensitivity", "e8": "E8 efficiency"}


def run_one(name: str, cfg, seeds: list[int], outdir: Path, smoke: bool,
            epochs: int) -> list[dict]:
    log = get_logger("exp")
    log.info(f"== {TITLES[name]} seeds={seeds} smoke={smoke} ==")
    mod = EXPS[name]
    import inspect

    C.ensure_defaults(cfg)
    kw = dict(seeds=seeds, outdir=str(outdir / name), smoke=smoke)
    if "epochs" in inspect.signature(mod.run).parameters:
        kw["epochs"] = epochs
    rows = mod.run(cfg, **kw)
    C.save_all(rows, str(outdir / f"{name}{'_smoke' if smoke else ''}"))
    if name in ("e1", "e3", "e4", "e5", "e7") and rows and "strict_f1" in rows[0]:
        def _label(r):
            parts = [str(r[k]) for k in ("harmful_kind", "condition", "pair", "ablation",
                                         "alpha", "tau", "model") if k in r]
            return "|".join(parts)
        cats, series = [], {}
        for r in rows:
            lab = _label(r)
            if lab not in cats:
                cats.append(lab)
            series.setdefault(f"seed{r.get('seed', 0)}", []).append(float(r["strict_f1"]))
        C.barplot(cats, series, str(outdir / "figs" / f"{name}_f1.png"), TITLES[name])
    log.info(f"{name}: {len(rows)} rows")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--opt", nargs="*", default=[])
    ap.add_argument("--exp", default=None, choices=sorted(EXPS))
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--all-smoke", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--outdir", default="reports/exp")
    args = ap.parse_args()
    cfg = load_config(args.config, args.opt)
    smoke = args.all_smoke or args.config.endswith("smoke.yaml")
    seeds = args.seeds or ([0] if smoke else [0, 1, 2, 3, 4])
    epochs = 3 if smoke else 5
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figs").mkdir(exist_ok=True)
    names = [args.exp] if args.exp else sorted(EXPS)
    all_rows = {}
    for n in names:
        all_rows[n] = run_one(n, cfg, seeds, outdir, smoke, epochs)
    print(f"done: { {k: len(v) for k, v in all_rows.items()} } -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
