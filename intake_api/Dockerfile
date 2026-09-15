# Microservice 1 — Intake API   (single-file edition)
#
# Build from THIS folder (intake_api/):
#   docker build -t intake-api:latest .
#
# This folder contains only the Intake API's files, so the build context is
# already scoped to this service.
FROM python:3.12-slim

WORKDIR /service

# Dependencies first (layer caching). The Intake API does not need
# scikit-learn or joblib — it only calls the ML API over HTTP.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy this service's code + its dataset.
COPY security.py .
COPY intake_api.py .
COPY food.csv .

EXPOSE 8001

# ML_API_URL is injected per-environment by the k8s manifest. The default
# below points at the in-cluster ml-api Service.
ENV ML_API_URL="http://ml-api:8002"

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8001/health').status_code==200 else 1)"

CMD ["uvicorn", "intake_api:app", "--host", "0.0.0.0", "--port", "8001"]
