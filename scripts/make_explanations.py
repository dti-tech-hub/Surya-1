"""Phase-5 explainer script: top-10 detected anomalies -> gallery + explainer metrics table.

Trains source-only contrastive (shared helper), detects on target test, explains each flagged window
with prototype/attention/attribution(IG+saliency)/counterfactual/cross-domain/NL parts.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import numpy as np
from tsad.config import load_config
from tsad.seed import set_seed
from tsad.logging_utils import get_logger
from tsad.data.windows import window_label
from scripts.train import prepare_and_train
from tsad.explain.base import new_explanation, topk_channels
from tsad.explain.prototype_exp import prototype_part
from tsad.explain.attention_exp import attention_part
from tsad.explain.attribution import SoftProtoScore, ATTRIBUTORS
from tsad.explain.counterfactual import counterfactual
from tsad.explain.cross_domain import cross_domain_part
from tsad.explain.nl_summary import summarize
from tsad.explain import metrics as em
from tsad.explain.gallery import anomaly_panel, index_html
from tsad.eval.report import write_table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--opt", nargs="*", default=[])
    ap.add_argument("--out", default="reports/explain")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    cfg = load_config(args.config, args.opt)
    for k, v in {"encoder": "tcn", "embedding_dim": 8, "num_prototypes": 3}.items():
        cfg.model.setdefault(k, v)
    if "loss" not in cfg:
        cfg.loss = {"tau": 0.1, "w_nce": 1.0, "w_ts2vec": 0.0, "w_mask": 0.0}
    if "augment" not in cfg:
        cfg.augment = ["jitter"]
    set_seed(int(cfg.seed))
    log = get_logger("explain")
    art = prepare_and_train(cfg)
    mod, bank, W, S = art["mod"], art["bank"], art["W"], art["S"]
    te, te_w, tel, starts, thr = art["te"], art["te_w"], art["tel"], art["starts"], art["thr"]
    te_wl = window_label(tel, W, S)
    gt_mask = art["d"].test_channel_mask  # (Tte, C) point-level GT anomalous channels
    # detection order: top window scores
    Zte = art["Zte"]
    win_scores = bank.score(Zte)
    order = np.argsort(-win_scores)[: args.top]
    score_fn = SoftProtoScore(mod.enc, bank.prototypes)
    Ztr, _ = mod.embed(np.stack([art["te_w"][0]]))  # warmup (batchnorm-safe)
    # normal train windows for medoids: reuse test-normal windows as reference pool
    ref_pool, ref_embs = te_w[te_wl == 0], Zte[te_wl == 0]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items, metric_rows = [], []
    for rank, wi in enumerate(order):
        s, t0 = starts[wi], starts[wi]
        exp = new_explanation(int(wi), t0, t0 + W, float(win_scores[wi]), float(thr))
        x = te_w[wi]
        exp["parts"]["prototype"] = prototype_part(bank, Zte[wi], ref_pool, ref_embs)
        _, attn = mod.embed(te_w[wi:wi + 1])
        exp["parts"]["attention"] = attention_part(None if attn is None else attn[0])
        attrs = {n: fn(score_fn, x) for n, fn in ATTRIBUTORS.items()}
        exp["parts"]["attribution"] = {
            "method": "ig", "top_channels": topk_channels(attrs["ig"]),
            "top_timesteps": [int(i) for i in np.argsort(-np.abs(attrs["ig"]).sum(1))[:5]],
        }
        exp["parts"]["counterfactual"] = counterfactual(
            score_fn, x, float(thr), lo=float(te.min()), hi=float(te.max()))
        exp["parts"]["cross_domain"] = cross_domain_part(
            Zte[wi], {"domain_0": bank.prototypes}, None)
        exp["parts"]["summary"] = summarize(exp)
        med = np.asarray(exp["parts"]["prototype"]["medoid_window"])
        panel = anomaly_panel(x, med, attrs["ig"], out / f"anomaly_{rank}.png",
                              f"#{wi} t={t0}-{t0 + W} score={win_scores[wi]:.2f}")
        items.append({"window_id": int(wi), "t_start": t0, "t_end": t0 + W,
                      "score": float(win_scores[wi]), "panel": str(panel),
                      "summary": exp["parts"]["summary"]})
        (out / f"anomaly_{rank}.json").write_text(json.dumps(exp, indent=1))
        # metrics per explainer
        mw = gt_mask[t0:t0 + W]  # (W, C) point-level GT for this window
        for n, a in attrs.items():
            assert a.shape == x.shape, (n, a.shape, x.shape)
            loc = em.localization(a, mw)
            metric_rows.append({"window": int(wi), "explainer": n,
                                "deletion_auc": round(em.deletion_auc(score_fn, x, a, med), 4),
                                "loc_p_at_k": round(loc["precision_at_k"], 4),
                                "loc_auprc": round(float(loc["auprc"]), 4) if loc["auprc"] == loc["auprc"] else "nan"})
        cf = exp["parts"]["counterfactual"]
        metric_rows.append({"window": int(wi), "explainer": "counterfactual",
                            "deletion_auc": "n/a", "loc_p_at_k": "n/a", "loc_auprc": "n/a",
                            "cf_valid": cf["valid"], "cf_sparsity": round(1 - cf["frac_changed"], 4),
                            "cf_l1": round(cf["delta_l1"], 4)})
        log.info(f"#{wi}: score={win_scores[wi]:.2f} ch={exp['parts']['attribution']['top_channels']}")
    index_html(items, out / "index.html")
    write_table([r for r in metric_rows if r["explainer"] != "counterfactual"], str(out / "_metrics"))
    cf_rows = [r for r in metric_rows if r["explainer"] == "counterfactual"]
    write_table([{"window": r["window"], "valid": r["cf_valid"], "sparsity": r["cf_sparsity"],
                  "l1": r["cf_l1"]} for r in cf_rows], str(out / "_cf_metrics"))
    print(f"gallery: {len(items)} anomalies -> {out}/index.html + metrics tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
