"""Phase-7 API tests (FastAPI TestClient + httpx semantics). Needs models/bundle/synthetic_tcn."""

import os
import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tsad.api.app import app


@pytest.fixture(scope="module", autouse=True)
def _isolated_bundles(tmp_path_factory):
    """Copy the built bundle to tmp; point TSAD_BUNDLES there so /adapt writes don't pollute the repo."""
    tmp = tmp_path_factory.mktemp("bundles")
    shutil.copytree("models/bundle/synthetic_tcn", tmp / "synthetic_tcn")
    os.environ["TSAD_BUNDLES"] = str(tmp)
    import tsad.api.app as appmod

    appmod._BUNDLES.clear()
    yield
    appmod._BUNDLES.clear()


client = TestClient(app)


def _series(T: int = 300, C: int = 2, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    x = np.sin(np.linspace(0, 20, T))[:, None] * np.ones((1, C)) + 0.1 * rng.standard_normal((T, C))
    x[100, 0] += 6.0  # spike
    return x.tolist()


def test_health_and_models():
    assert client.get("/health").json() == {"status": "ok"}
    assert "synthetic_tcn" in client.get("/models").json()["models"]


def test_detect_json_and_csv():
    r = client.post("/detect", json={"series": _series(), "model": "synthetic_tcn"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["scores"]) == 300 and len(body["flags"]) == 300
    assert body["n_flagged"] >= 0 and "threshold" in body
    import io

    buf = io.StringIO()
    np.savetxt(buf, np.array(_series()), delimiter=",")
    files = {"file": ("s.csv", buf.getvalue(), "text/csv")}
    r2 = client.post("/detect_csv?model=synthetic_tcn", files=files)
    assert r2.status_code == 200, r2.text
    assert r2.json()["T"] == 300


def test_detect_bad_shape():
    r = client.post("/detect", json={"series": [[1.0]], "model": "synthetic_tcn"})
    assert r.status_code == 400  # shorter than window


def test_explain_schema():
    r = client.post("/explain", json={"series": _series(), "model": "synthetic_tcn"})
    assert r.status_code == 200, r.text
    parts = r.json()["parts"]
    assert {
        "prototype",
        "attention",
        "attribution",
        "counterfactual",
        "cross_domain",
        "summary",
    } <= set(parts)
    assert isinstance(parts["summary"], str) and len(parts["summary"]) > 20


def test_domains_and_adapt():
    assert client.get("/domains?model=synthetic_tcn").status_code == 200
    import io

    x = np.array(_series(T=400))
    buf = io.StringIO()
    np.savetxt(buf, x, delimiter=",")
    r = client.post(
        "/adapt?model=synthetic_tcn&new_name=test_adapted",
        files={"file": ("s.csv", buf.getvalue(), "text/csv")},
    )
    assert r.status_code == 200, r.text
    assert "test_adapted" in client.get("/models").json()["models"]
    r2 = client.post("/detect", json={"series": _series(), "model": "test_adapted"})
    assert r2.status_code == 200
