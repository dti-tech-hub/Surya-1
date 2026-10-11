# Makefile — Windows PowerShell (5.1) compatible via python invocations.
# All targets use `python -m ...` so they work with or without `make` on PATH.
PY = python

setup:
	$(PY) -m pip install -r requirements-smoke.txt

setup-full:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts experiments || exit 0
	$(PY) -m black --check src tests scripts experiments || exit 0

smoke:
	$(PY) scripts/smoke.py --config configs/smoke.yaml

train:
	$(PY) scripts/train.py --config configs/default.yaml $(ARGS)

eval:
	$(PY) scripts/eval.py --config configs/default.yaml $(ARGS)

explain:
	$(PY) scripts/make_explanations.py --config configs/default.yaml $(ARGS)

experiments-smoke:
	$(PY) scripts/run_experiment.py --config configs/smoke.yaml --all-smoke

experiments:
	$(PY) scripts/run_experiment.py --config configs/default.yaml --all

app:
	$(PY) -m uvicorn tsad.api.app:app --app-dir src --host 0.0.0.0 --port 8000

dashboard:
	$(PY) -m streamlit run dashboard/app_streamlit.py

bundle:
	$(PY) scripts/build_bundle.py --config configs/smoke.yaml

report:
	$(PY) scripts/make_report.py --config configs/default.yaml
