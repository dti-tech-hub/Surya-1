"""Developer helper: start the inference API (and optionally the dashboard).

Usage:
    python scripts/serve.py                 # API on :8000
    python scripts/serve.py --port 8010     # API on another port
    python scripts/serve.py --api-only      # skip the dashboard
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def wait(url: str, timeout: float = 90.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            return requests.get(url, timeout=2).status_code == 200
        except requests.RequestException:
            time.sleep(0.5)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dashboard-port", type=int, default=8501)
    ap.add_argument("--api-only", action="store_true")
    ap.add_argument("--model", default="synthetic_tcn")
    a = ap.parse_args()

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["TSAD_BUNDLES"] = str(ROOT / "models" / "bundle")

    api = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "tsad.api.app:app",
            "--app-dir",
            str(ROOT / "src"),
            "--host",
            a.host,
            "--port",
            str(a.port),
        ],
        cwd=ROOT,
        env=env,
    )
    base = f"http://{a.host}:{a.port}"
    if not wait(f"{base}/health"):
        api.terminate()
        print(
            f"API did not come up on {base}; run `python scripts/build_bundle.py` first.",
            file=sys.stderr,
        )
        return 1
    print(f"API      -> {base}  (docs {base}/docs)")
    print(f"models   -> {requests.get(f'{base}/models', timeout=5).json()}")

    dash = None
    if not a.api_only:
        dash = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "dashboard/app_streamlit.py",
                "--server.port",
                str(a.dashboard_port),
                "--server.headless",
                "true",
            ],
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if wait(f"http://127.0.0.1:{a.dashboard_port}/_stcore/health", timeout=60):
            print(f"Dashboard-> http://localhost:{a.dashboard_port}")
        else:
            print(f"Dashboard failed to start on {a.dashboard_port}", file=sys.stderr)

    try:
        api.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in (api, dash):
            if p is not None and p.poll() is None:
                p.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
