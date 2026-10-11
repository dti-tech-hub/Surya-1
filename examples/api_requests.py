"""Example API requests (run server first: make app). Uses synthetic_tcn bundle."""
import json
import numpy as np
import httpx

BASE = "http://localhost:8000"
rng = np.random.default_rng(0)
T, C = 300, 2
x = np.sin(np.linspace(0, 20, T))[:, None] * np.ones((1, C)) + 0.1 * rng.standard_normal((T, C))
x[100, 0] += 6.0

print("health:", httpx.get(f"{BASE}/health").json())
print("models:", httpx.get(f"{BASE}/models").json())
det = httpx.post(f"{BASE}/detect", json={"series": x.tolist(), "model": "synthetic_tcn"}).json()
print("detect: n_flagged =", det["n_flagged"], "thr =", round(det["threshold"], 3))
exp = httpx.post(f"{BASE}/explain", json={"series": x.tolist(), "model": "synthetic_tcn"}).json()
print("explain:", exp["parts"]["summary"])
print("domains:", json.dumps(httpx.get(f"{BASE}/domains?model=synthetic_tcn").json())[:200])
