FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Bake a synthetic demo bundle at build time (CPU, ~2 min) so `compose up` serves instantly.
RUN python scripts/build_bundle.py --config configs/smoke.yaml --name synthetic_tcn
EXPOSE 8000 8501
CMD ["python", "-m", "uvicorn", "tsad.api.app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
