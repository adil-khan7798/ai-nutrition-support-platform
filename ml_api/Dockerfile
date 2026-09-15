# Microservice 2 — ML API   (single-file edition)
#
# Build from THIS folder (ml_api/):
#   docker build -t ml-api:latest .
#
# This folder contains only the ML API's files, so the build context is
# already scoped to this service.
FROM python:3.12-slim

WORKDIR /service

# Dependencies first (layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy this service's code + its dataset.
COPY security.py .
COPY ml_api.py .
COPY nhanes.csv .

# Train the models at build time from the bundled NHANES dataset, so the
# image is self-contained and starts fast. Writes ./models/*.joblib +
# models/metadata.json next to ml_api.py.
RUN python ml_api.py train

EXPOSE 8002

# Liveness/readiness probe target for AKS: GET /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8002/health').status_code==200 else 1)"

# Serve with uvicorn. Models already exist from the build step above.
CMD ["uvicorn", "ml_api:app", "--host", "0.0.0.0", "--port", "8002"]
