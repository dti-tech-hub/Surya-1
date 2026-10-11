"""FastAPI inference service: /health /models /detect /explain /domains /adapt."""
from __future__ import annotations
import io
import json
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from tsad.api.registry import Bundle

app = FastAPI(title="TSAD adaptive anomaly detection", version="0.1.0")
_BUNDLES: dict[str, Bundle] = {}


def bundle_dir() -> Path:
    import os

    return Path(os.environ.get("TSAD_BUNDLES", "models/bundle"))


def get_bundle(name: str) -> Bundle:
    if name not in _BUNDLES:
        p = bundle_dir() / name
        if not (p / "stats.json").exists():
            raise HTTPException(404, f"bundle '{name}' not found; POST /adapt or run build_bundle.py")
        _BUNDLES[name] = Bundle(p)
    return _BUNDLES[name]


class DetectJSON(BaseModel):
    series: list[list[float]] = Field(description="(T, C) raw series")
    model: str = "synthetic_tcn"
    domain_id: str = "domain_0"


class ExplainReq(BaseModel):
    series: list[list[float]]
    model: str = "synthetic_tcn"
    t_start: int | None = None  # focus window start; default = top-score window


def _csv_to_array(raw: bytes) -> np.ndarray:
    try:
        return np.loadtxt(io.BytesIO(raw), delimiter=",", dtype=float)
    except Exception:
        pass
    try:
        import pandas as pd

        return pd.read_csv(io.BytesIO(raw), header=None).to_numpy(dtype=float)
    except Exception as e:
        raise HTTPException(400, f"CSV parse failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    d = bundle_dir()
    return {"models": sorted(p.name for p in d.iterdir() if (p / "stats.json").exists()) if d.exists() else []}


@app.get("/domains")
def domains(model: str = "synthetic_tcn"):
    b = get_bundle(model)
    return {"model": model, "n_channels": b.stats["n_ch"], "window": b.stats["W"],
            "stride": b.stats["S"], "threshold": b.stats["threshold"],
            "note": "per-domain stats come from data_summary; bundle carries train-domain norm stats",
            "norm_mean": b.stats["norm_mean"], "norm_std": b.stats["norm_std"]}


@app.post("/detect")
def detect(req: DetectJSON):
    b = get_bundle(req.model)
    try:
        return {"model": req.model, "domain_id": req.domain_id, **b.detect(np.asarray(req.series))}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/detect_csv")
async def detect_csv(model: str = "synthetic_tcn", domain_id: str = "domain_0",
                     file: UploadFile = File(...)):
    b = get_bundle(model)
    x = _csv_to_array(await file.read())
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    try:
        return {"model": model, "domain_id": domain_id, **b.detect(x)}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/explain")
def explain(req: ExplainReq):
    import torch
    from tsad.data.windows import sliding_windows
    from tsad.explain.base import new_explanation, topk_channels
    from tsad.explain.prototype_exp import prototype_part
    from tsad.explain.attention_exp import attention_part
    from tsad.explain.attribution import SoftProtoScore, integrated_gradients
    from tsad.explain.counterfactual import counterfactual
    from tsad.explain.cross_domain import cross_domain_part
    from tsad.explain.nl_summary import summarize

    b = get_bundle(req.model)
    det = b.detect(np.asarray(req.series))
    W, S = b.stats["W"], b.stats["S"]
    xn = b.normalize(np.asarray(req.series, dtype=float))
    w = sliding_windows(xn, W, S)
    with torch.no_grad():
        Z = np.concatenate([b.enc(torch.tensor(w[i:i + 256]))["rep"].numpy()
                            for i in range(0, len(w), 256)])
    scores = b.bank.score(Z)
    wi = int(np.argmax(scores)) if req.t_start is None else max(
        0, min(len(w) - 1, (req.t_start) // max(1, S)))
    t0 = wi * S
    fn = SoftProtoScore(b.enc, b.bank.prototypes)
    x = w[wi]
    exp = new_explanation(wi, t0, t0 + W, float(det["scores"][t0]), det["threshold"])
    exp["parts"]["prototype"] = prototype_part(b.bank, Z[wi], w, Z)
    try:  # attention weights only exist for transformer encoders
        with torch.no_grad():
            a = b.enc(torch.tensor(w[wi:wi + 1]))["attn"]
        exp["parts"]["attention"] = attention_part(None if a is None else a[0].numpy())
    except Exception:
        exp["parts"]["attention"] = attention_part(None)
    attr = integrated_gradients(fn, x, steps=10)
    exp["parts"]["attribution"] = {"method": "ig", "top_channels": topk_channels(attr)}
    exp["parts"]["counterfactual"] = counterfactual(fn, x, float(det["threshold"]),
                                                    lo=float(xn.min()), hi=float(xn.max()),
                                                    steps=50)
    exp["parts"]["cross_domain"] = cross_domain_part(Z[wi], {"train": b.bank.prototypes}, None)
    exp["parts"]["summary"] = summarize(exp)
    return json.loads(json.dumps(exp))  # ensure JSON-serializable


@app.post("/adapt")
async def adapt(model: str = "synthetic_tcn", new_name: str = "adapted",
                file: UploadFile = File(...)):
    """Target-time adaptation (prototype refresh): fit new norm + prototype bank on unlabeled CSV.

    Encoder weights are reused (no retraining); threshold recalibrated on the new data's own
    score distribution (95th percentile). Returns the new bundle name.
    """
    import torch
    from tsad.data.windows import sliding_windows

    b = get_bundle(model)
    x = _csv_to_array(await file.read())
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape[1] != b.stats["n_ch"] or len(x) < b.stats["W"] * 2:
        raise HTTPException(400, f"need (T>={2 * b.stats['W']}, {b.stats['n_ch']})")
    mu, sd = x.mean(0), np.where(x.std(0) < 1e-8, 1.0, x.std(0))
    xn = ((x - mu) / sd).astype(np.float32)
    W, S = b.stats["W"], b.stats["S"]
    w = sliding_windows(xn, W, S)
    with torch.no_grad():
        Z = np.concatenate([b.enc(torch.tensor(w[i:i + 256]))["rep"].numpy()
                            for i in range(0, len(w), 256)])
    from sklearn.cluster import KMeans

    k = min(8, len(Z))
    proto = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z).cluster_centers_
    from tsad.models.scoring import map_to_timestamps, smooth

    D = np.linalg.norm(Z[:, None, :] - proto[None, :, :], axis=2).min(1)
    ts = smooth(map_to_timestamps(D, list(range(0, len(xn) - W + 1, S)), len(xn), W), 5)
    thr = float(np.quantile(ts, 0.95))
    out = bundle_dir() / new_name
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": b.enc.state_dict(), "kind": "tcn", "emb_d": Z.shape[1]}, out / "encoder.pt")
    np.save(out / "prototypes.npy", proto)
    (out / "stats.json").write_text(json.dumps(
        {"norm_mean": mu.tolist(), "norm_std": sd.tolist(), "cal_mu": 0.0, "cal_sd": 1.0,
         "threshold": thr, "W": W, "S": S, "smooth_width": 5, "n_ch": x.shape[1],
         "note": f"adapted from {model} (prototype refresh, unlabeled)"}, indent=1))
    (out / "config.yaml").write_text(f"# adapted from {model}\n")
    _BUNDLES.pop(new_name, None)
    return {"new_model": new_name, "threshold": thr, "n_windows": len(w)}
