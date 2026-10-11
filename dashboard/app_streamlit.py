"""Streamlit dashboard: upload series -> scores + flags -> per-anomaly explanation. No server needed."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import streamlit as st
import torch

from tsad.api.registry import Bundle
from tsad.data.windows import sliding_windows
from tsad.explain.attribution import SoftProtoScore, integrated_gradients
from tsad.explain.base import new_explanation, topk_channels
from tsad.explain.counterfactual import counterfactual
from tsad.explain.nl_summary import summarize
from tsad.explain.prototype_exp import prototype_part

st.set_page_config(page_title="TSAD demo", layout="wide")
st.title("Adaptive domain-aware TSAD — demo")


@st.cache_resource
def load_bundle(name: str) -> Bundle:
    return Bundle(Path("models/bundle") / name)


models = (
    sorted(p.name for p in Path("models/bundle").iterdir() if (p / "stats.json").exists())
    if Path("models/bundle").exists()
    else []
)
if not models:
    st.error("No bundles. Run: python scripts/build_bundle.py --config configs/smoke.yaml")
    st.stop()
name = st.sidebar.selectbox("model", models)
b = load_bundle(name)
up = st.file_uploader("CSV series (T rows, C columns, no header)", type=["csv"])
if up is None:
    st.info("Upload a CSV to begin (or generate one via scripts/make_synthetic.py).")
    st.stop()
x = np.loadtxt(up, delimiter=",", dtype=float)
if x.ndim == 1:
    x = x.reshape(-1, 1)
det = b.detect(x)
thr = st.slider(
    "threshold", float(np.min(det["scores"])), float(np.max(det["scores"])), float(det["threshold"])
)
flags = (np.array(det["scores"]) >= thr).astype(int)
st.line_chart({"score": det["scores"]})
st.write(f"flagged {int(flags.sum())}/{len(flags)} @ {thr:.2f} (bundle thr {det['threshold']:.2f})")
anom = np.where(flags == 1)[0]
if not len(anom):
    st.success("No anomalies at this threshold.")
    st.stop()
t0 = st.selectbox("inspect timestep", [int(a) for a in anom[:50]])
W, S = b.stats["W"], b.stats["S"]
wi = max(0, min(len(sliding_windows(b.normalize(x), W, S)) - 1, (t0 - W // 2) // S))
xn = b.normalize(x)
w = sliding_windows(xn, W, S)
with torch.no_grad():
    Z = np.concatenate(
        [b.enc(torch.tensor(w[i : i + 256]))["rep"].numpy() for i in range(0, len(w), 256)]
    )
fn = SoftProtoScore(b.enc, b.bank.prototypes)
exp = new_explanation(wi, wi * S, wi * S + W, float(det["scores"][t0]), float(thr))
exp["parts"]["prototype"] = prototype_part(b.bank, Z[wi], w, Z)
attr = integrated_gradients(fn, w[wi], steps=10)
exp["parts"]["attribution"] = {"top_channels": topk_channels(attr)}
exp["parts"]["counterfactual"] = counterfactual(
    fn, w[wi], float(thr), lo=float(xn.min()), hi=float(xn.max()), steps=50
)
exp["parts"]["summary"] = summarize(
    {**exp, "parts": {**exp["parts"], "attention": {}, "cross_domain": {}}}
)
st.subheader("Explanation")
st.write(exp["parts"]["summary"])
st.bar_chart({f"ch{c}": float(v) for c, v in enumerate(np.abs(attr).mean(0))})
st.write("counterfactual valid:", exp["parts"]["counterfactual"]["valid"])
