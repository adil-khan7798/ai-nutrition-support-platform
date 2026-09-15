"""
ml_api.py — Microservice 2: ML API (single-file edition).

A self-contained FastAPI inference service. All logic that was previously
split across app/main.py, app/schemas.py, app/model_service.py and
shared/feature_spec.py is collapsed into this one file.

Receives the demographic profile + aggregated nutrient totals computed by the
Intake API and returns disease risk predictions for hypertension,
hypercholesterolemia, type 2 diabetes, and GERD.

--------------------------------------------------------------------------
DATA / MODEL FILES (must sit next to this file, or set env vars):
  nhanes.csv  ........  NHANES 2013-2014 training data   (env: NHANES_CSV)
  models/     ........  directory for trained .joblib files (env: MODEL_DIR)
--------------------------------------------------------------------------

USAGE
  Train the models (first run, or whenever data changes):
      python ml_api.py train

  Run the API (auto-trains if models are missing):
      python ml_api.py serve            # or just: python ml_api.py
      # -> http://localhost:8002/docs

  Run with uvicorn directly (models must already be trained):
      uvicorn ml_api:app --host 0.0.0.0 --port 8002

ENDPOINTS (public)
  GET  /          — service banner
  GET  /health    — liveness probe
  GET  /ready     — readiness probe

ENDPOINTS (protected — requires Authorization: Bearer <token>)
  GET  /metadata  — training metadata         [user, admin, service]
  POST /predict   — run inference             [service, admin]
"""

import datetime
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

from security import peek_token, role_required

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml_api")

# ===========================================================================
# CONFIG / PATHS
# ===========================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
NHANES_CSV = os.environ.get("NHANES_CSV", os.path.join(HERE, "nhanes.csv"))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(HERE, "models"))
PORT = int(os.environ.get("PORT", "8002"))

DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
DATABRICKS_ENDPOINT_URL = os.environ.get(
    "DATABRICKS_ENDPOINT_URL",
    "https://<your-workspace>.cloud.databricks.com/serving-endpoints/health-food-risk-endpoint/invocations",
)

# All lab columns the Databricks Spark ML pipeline was trained on. Every key
# must be present in the request (value can be None — the Imputer fills it with
# the training median). Omitting a key entirely causes a FIELD_NOT_FOUND error.
_DATABRICKS_LAB_COLS = [
    "lab_URXUMA",
    "lab_URXUMS",
    "lab_URXUCR_x",
    "lab_URXCRS",
    "lab_URDACT",
    "lab_LBXSAL",
    "lab_LBDSALSI",
    "lab_LBXSAPSI",
    "lab_LBXSASSI",
    "lab_LBXSATSI",
    "lab_LBXSBU",
    "lab_LBDSBUSI",
    "lab_LBXSC3SI",
    "lab_LBXSCA",
    "lab_LBDSCASI",
    "lab_LBXSCH",
    "lab_LBDSCHSI",
    "lab_LBXSCK",
    "lab_LBXSCLSI",
    "lab_LBXSCR",
    "lab_LBDSCRSI",
    "lab_LBXSGB",
    "lab_LBDSGBSI",
    "lab_LBXSGL",
    "lab_LBDSGLSI",
    "lab_LBXSGTSI",
    "lab_LBXSIR",
    "lab_LBDSIRSI",
    "lab_LBXSKSI",
    "lab_LBXSLDSI",
    "lab_LBXSNASI",
    "lab_LBXSOSSI",
    "lab_LBXSPH",
    "lab_LBDSPHSI",
    "lab_LBXSTB",
    "lab_LBDSTBSI",
    "lab_LBXSTP",
    "lab_LBDSTPSI",
    "lab_LBXSTR",
    "lab_LBDSTRSI",
    "lab_LBXSUA",
    "lab_LBDSUASI",
    "lab_LBXWBCSI",
    "lab_LBXLYPCT",
    "lab_LBXMOPCT",
    "lab_LBXNEPCT",
    "lab_LBXEOPCT",
    "lab_LBXBAPCT",
    "lab_LBDLYMNO",
    "lab_LBDMONO",
    "lab_LBDNENO",
    "lab_LBDEONO",
    "lab_LBDBANO",
    "lab_LBXRBCSI",
    "lab_LBXHGB",
    "lab_LBXHCT",
    "lab_LBXMCVSI",
    "lab_LBXMCHSI",
    "lab_LBXMC",
    "lab_LBXRDW",
    "lab_LBXPLTSI",
    "lab_LBXMPSI",
    "lab_PHQ020",
    "lab_PHQ030",
    "lab_PHQ040",
    "lab_PHQ050",
    "lab_PHQ060",
    "lab_PHAFSTHR_x",
    "lab_PHAFSTMN_x",
    "lab_PHDSESN",
    "lab_LBDHDD",
    "lab_LBDHDDSI",
    "lab_LBXHA",
    "lab_LBXHBS",
    "lab_LBXHBC",
    "lab_LBDHBG",
    "lab_LBDHD",
    "lab_LBDHEG",
    "lab_LBDHEM",
    "lab_LBXGH",
    "lab_WTSH2YR_x",
    "lab_LBXTC",
    "lab_LBDTCSI",
    "lab_LBXTTG",
    "lab_WTSH2YR_y",
    "lab_URXVOL1",
    "lab_URDFLOW1",
]

# ===========================================================================
# FEATURE SPEC — the data contract shared with the Intake API.
# ---------------------------------------------------------------------------
# NUTRIENT_FIELDS is the VERIFIED intersection of the two datasets. The NHANES
# file also has `calories` and `caffeine_mg`, but the USDA food.csv has no
# energy/calorie column and no caffeine column, so those two cannot be sourced
# per-food. The honest common set is the 8 fields below.
# ===========================================================================
NUTRIENT_FIELDS = [
    "protein_g",
    "carbs_g",
    "sugar_g",
    "fiber_g",
    "total_fat_g",
    "saturated_fat_g",
    "cholesterol_mg",
    "sodium_mg",
]

# Exact column order fed to the models. MUST match training and serving.
# bmi is derived (weight_kg / (height_cm/100)^2).
FEATURE_ORDER = [
    "age",
    "gender",
    "race_ethnicity",
    "education_level",
    "weight_kg",
    "height_cm",
    "bmi",
] + NUTRIENT_FIELDS

DISEASES = ["hypertension", "hypercholesterolemia", "type_2_diabetes", "gerd"]

# Substring patterns used to derive a positive label from the NHANES
# free-text `conditions_list` column. Simple, auditable text matches.
DISEASE_PATTERNS = {
    "hypertension": ["hypertension"],
    "hypercholesterolemia": ["hypercholesterol", "cholesterolemia"],
    "type_2_diabetes": ["type 2 diabetes"],
    "gerd": ["gastro-esophageal reflux", "gerd"],
}


# ===========================================================================
# SCHEMAS (Pydantic request/response models)
# ===========================================================================
class PredictRequest(BaseModel):
    """Input to POST /predict — demographic profile + aggregated nutrient totals.

    Field names and codings match the NHANES dataset the models were trained on.
    """

    age: float = Field(..., ge=0, le=120, description="Age in years")
    gender: int = Field(..., ge=1, le=2, description="NHANES coding: 1=male, 2=female")
    race_ethnicity: int = Field(..., ge=1, le=5, description="NHANES coding 1-5")
    education_level: int = Field(..., ge=1, le=5, description="NHANES coding 1-5")
    weight_kg: float = Field(..., gt=0, le=400)
    height_cm: float = Field(..., gt=0, le=260)

    protein_g: float = Field(..., ge=0)
    carbs_g: float = Field(..., ge=0)
    sugar_g: float = Field(..., ge=0)
    fiber_g: float = Field(..., ge=0)
    total_fat_g: float = Field(..., ge=0)
    saturated_fat_g: float = Field(..., ge=0)
    cholesterol_mg: float = Field(..., ge=0)
    sodium_mg: float = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 54,
                "gender": 1,
                "race_ethnicity": 3,
                "education_level": 3,
                "weight_kg": 89.5,
                "height_cm": 176.8,
                "protein_g": 95.2,
                "carbs_g": 240.1,
                "sugar_g": 88.4,
                "fiber_g": 18.0,
                "total_fat_g": 70.5,
                "saturated_fat_g": 24.3,
                "cholesterol_mg": 310.0,
                "sodium_mg": 3400.0,
            }
        }
    }


class DiseaseFlag(BaseModel):
    flag: bool = Field(..., description="True if predicted probability >= 0.5")
    probability: float = Field(..., ge=0, le=1)


class PredictResponse(BaseModel):
    overall_health_risk: bool = Field(
        ..., description="True if any individual disease flag is True"
    )
    overall_probability: float = Field(
        ..., ge=0, le=1, description="Mean of the four disease probabilities"
    )
    disease_flags: Dict[str, DiseaseFlag]


# ===========================================================================
# TRAINING — produces models/<disease>.joblib + models/metadata.json
# ===========================================================================
def train_models() -> None:
    """Train four RandomForest classifiers on the real NHANES dataset.

    The project brief also permits Spark ML; the feature contract here is
    identical, so a Spark ML training job can replace this function without
    changing the serving code.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    if not os.path.exists(NHANES_CSV):
        raise FileNotFoundError(f"NHANES data not found at {NHANES_CSV}")

    df = pd.read_csv(NHANES_CSV)
    n_patients = len(df)
    logger.info(
        "Loaded NHANES dataset: %d patients, %d columns", n_patients, df.shape[1]
    )

    # Derive 0/1 labels from the conditions_list text column.
    conditions = df["conditions_list"].fillna("").str.lower()
    labels = {}
    for disease, patterns in DISEASE_PATTERNS.items():
        match = pd.Series(False, index=df.index)
        for p in patterns:
            match = match | conditions.str.contains(p, regex=False)
        labels[disease] = match.astype(int)

    # Build the feature matrix; median-impute missing values.
    X = df[FEATURE_ORDER].apply(pd.to_numeric, errors="coerce")
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    os.makedirs(MODEL_DIR, exist_ok=True)
    metadata = {
        "training_dataset": f"NHANES 2013-2014 ({os.path.basename(NHANES_CSV)})",
        "n_patients": n_patients,
        "feature_order": FEATURE_ORDER,
        "nutrient_fields": NUTRIENT_FIELDS,
        "diseases": DISEASES,
        "model_type": "sklearn.ensemble.RandomForestClassifier",
        "per_disease": {},
        "feature_medians": {k: float(v) for k, v in medians.items()},
    }

    for disease in DISEASES:
        y = labels[disease]
        positives = int(y.sum())
        if positives < 5:
            logger.warning("[skip] %s: only %d positive cases", disease, positives)
            continue

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_tr, y_tr)

        proba = clf.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        try:
            auc = float(roc_auc_score(y_te, proba))
        except ValueError:
            auc = None
        acc = float(accuracy_score(y_te, pred))

        joblib.dump(clf, os.path.join(MODEL_DIR, f"{disease}.joblib"))
        metadata["per_disease"][disease] = {
            "positive_cases": positives,
            "prevalence": round(positives / n_patients, 4),
            "test_auc": round(auc, 4) if auc is not None else None,
            "test_accuracy": round(acc, 4),
        }
        auc_str = f"{auc:.3f}" if auc is not None else "n/a"
        logger.info("[ok] %s: pos=%d AUC=%s acc=%.3f", disease, positives, auc_str, acc)

    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(
        "Saved %d models + metadata.json to %s", len(metadata["per_disease"]), MODEL_DIR
    )


# ===========================================================================
# MODEL REGISTRY — loads models and runs inference
# ===========================================================================
class ModelRegistry:
    def __init__(self) -> None:
        self.models: Dict[str, object] = {}
        self.metadata: Dict = {}
        self.loaded = False

    def load(self) -> None:
        meta_path = os.path.join(MODEL_DIR, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.metadata = json.load(f)
        else:
            logger.warning("metadata.json not found in %s", MODEL_DIR)

        for disease in DISEASES:
            path = os.path.join(MODEL_DIR, f"{disease}.joblib")
            if os.path.exists(path):
                self.models[disease] = joblib.load(path)
                logger.info("Loaded model: %s", disease)
            else:
                logger.warning("Model file missing for '%s' (%s)", disease, path)
        self.loaded = True

    def _features_frame(self, req: PredictRequest) -> pd.DataFrame:
        """Single-row DataFrame in exact FEATURE_ORDER; bmi derived here."""
        height_m = req.height_cm / 100.0
        bmi = req.weight_kg / (height_m * height_m) if height_m > 0 else 0.0
        row = {
            "age": req.age,
            "gender": req.gender,
            "race_ethnicity": req.race_ethnicity,
            "education_level": req.education_level,
            "weight_kg": req.weight_kg,
            "height_cm": req.height_cm,
            "bmi": round(bmi, 2),
            "protein_g": req.protein_g,
            "carbs_g": req.carbs_g,
            "sugar_g": req.sugar_g,
            "fiber_g": req.fiber_g,
            "total_fat_g": req.total_fat_g,
            "saturated_fat_g": req.saturated_fat_g,
            "cholesterol_mg": req.cholesterol_mg,
            "sodium_mg": req.sodium_mg,
        }
        return pd.DataFrame([[row[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)

    def predict(self, req: PredictRequest) -> PredictResponse:
        if DATABRICKS_TOKEN:
            return _call_databricks_endpoint(req)

        X = self._features_frame(req)
        flags: Dict[str, DiseaseFlag] = {}
        probs = []
        for disease in DISEASES:
            model = self.models.get(disease)
            if model is None:
                # Missing model: report 0.0 rather than fabricate a number.
                flags[disease] = DiseaseFlag(flag=False, probability=0.0)
                continue
            proba = float(model.predict_proba(X)[0][1])
            flags[disease] = DiseaseFlag(flag=proba >= 0.5, probability=round(proba, 4))
            probs.append(proba)

        overall_prob = round(sum(probs) / len(probs), 4) if probs else 0.0
        overall_risk = any(f.flag for f in flags.values())
        return PredictResponse(
            overall_health_risk=overall_risk,
            overall_probability=overall_prob,
            disease_flags=flags,
        )


registry = ModelRegistry()


# ===========================================================================
# HELPERS
# ===========================================================================
def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _error_body(error: str, message: str, status_code: int, path: str) -> dict:
    return {
        "error": error,
        "message": message,
        "status_code": status_code,
        "path": path,
        "timestamp": _now_iso(),
    }


# ===========================================================================
# DATABRICKS INFERENCE
# ---------------------------------------------------------------------------
# When DATABRICKS_TOKEN is set the /predict route delegates to the hosted
# Databricks Model Serving endpoint instead of the local sklearn models.
#
# The endpoint is a Spark ML RandomForest pipeline (Imputer → VectorAssembler
# → RandomForestClassifier) registered in Unity Catalog via MLflow.
#
# Request: all 106 training columns must be present as keys. Columns not
# available from the user request (lab values, income_to_poverty_ratio, etc.)
# are sent as None — the Imputer fills them with training-set medians.
#
# Response: {"predictions": [0.0]}  — MLflow returns just the prediction
# column value (0.0 = lower risk, 1.0 = high risk). No probability is exposed.
# ===========================================================================
def _call_databricks_endpoint(req: PredictRequest) -> PredictResponse:
    """Send one inference request to the Databricks Model Serving endpoint."""
    record = {
        # Required fields
        "gender": req.gender,
        "age": float(req.age),
        "race_ethnicity": req.race_ethnicity,
        "education_level": req.education_level,
        "medication_count": 0,
        # Known nutrition/body fields from the intake request
        "weight_kg": req.weight_kg,
        "height_cm": req.height_cm,
        "calories": None,
        "protein_g": req.protein_g,
        "carbs_g": req.carbs_g,
        "sugar_g": req.sugar_g,
        "fiber_g": req.fiber_g,
        "total_fat_g": req.total_fat_g,
        "saturated_fat_g": req.saturated_fat_g,
        "cholesterol_mg": req.cholesterol_mg,
        "sodium_mg": req.sodium_mg,
        # Not available from the intake request — Imputer fills with medians
        "income_to_poverty_ratio": None,
        "waist_cm": None,
        "caffeine_mg": None,
        # All lab columns set to None — Imputer fills with training medians
        **{col: None for col in _DATABRICKS_LAB_COLS},
    }

    import httpx

    resp = httpx.post(
        DATABRICKS_ENDPOINT_URL,
        json={"dataframe_records": [record]},
        headers={
            "Authorization": f"Bearer {DATABRICKS_TOKEN}",
            "Content-Type": "application/json",
        },
        timeout=120.0,
    )
    resp.raise_for_status()

    # The endpoint returns {"predictions": [0.0]} — a single float per row.
    # 0.0 = lower risk, 1.0 = high risk. No probability score is available.
    raw = float(resp.json()["predictions"][0])
    is_high_risk = raw == 1.0
    flag = DiseaseFlag(flag=is_high_risk, probability=raw)
    return PredictResponse(
        overall_health_risk=is_high_risk,
        overall_probability=raw,
        disease_flags={
            "hypertension": flag,
            "hypercholesterolemia": flag,
            "type_2_diabetes": flag,
            "gerd": flag,
        },
    )


# ===========================================================================
# RATE LIMITER
# ===========================================================================
def _real_client_ip(request: Request) -> str:
    """Return the real client IP, reading X-Forwarded-For set by the ingress.

    Behind a Kubernetes ingress the direct connection is the proxy pod, not
    the user. X-Forwarded-For carries the original client IP as its first
    (leftmost) value; X-Real-IP is a single-value alternative set by nginx.
    Falls back to the raw connection address for local / direct traffic.
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.headers.get("X-Real-IP") or (
        request.client.host if request.client else "unknown"
    )


def _jwt_subject(request: Request) -> str:
    """Rate-limit key for service-to-service routes: JWT sub instead of IP.

    Internal pod-to-pod traffic has no X-Forwarded-For, so IP-based limiting
    collapses all callers into the same pod-IP bucket. Using the token subject
    gives each authenticated service identity its own independent bucket.
    Falls back to IP for unauthenticated requests.
    """
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = peek_token(auth[7:])
        if token and token.get("sub"):
            return f"jwt:{token['sub']}"
    return _real_client_ip(request)


limiter = Limiter(key_func=_real_client_ip)


# ===========================================================================
# FASTAPI APP
# ===========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-train if no models are present, so `python ml_api.py serve` just works.
    if not os.path.exists(os.path.join(MODEL_DIR, "metadata.json")):
        logger.info("No trained models found — training now...")
        train_models()
    registry.load()
    logger.info("ML API ready. Models loaded: %s", list(registry.models.keys()))
    yield


app = FastAPI(
    title="Disease Risk ML API",
    description="Microservice 2 — disease risk inference from diet + demographics.",
    version="2.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter


# ----- security headers + request logging -----
@app.middleware("http")
async def log_and_secure(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency_ms = round((time.time() - start) * 1000, 1)

    username = "-"
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        payload = peek_token(auth_header[7:])
        if payload:
            username = f"{payload.get('sub', '-')}({payload.get('role', '-')})"

    logger.info(
        "method=%s path=%s status=%d latency_ms=%.1f ip=%s user=%s",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
        _client_ip(request),
        username,
    )

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


# ----- exception handlers -----
@app.exception_handler(StarletteHTTPException)
async def http_exc_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    if isinstance(exc.detail, dict):
        content = exc.detail
    else:
        _labels: Dict[int, str] = {
            400: "Bad Request",
            401: "Unauthorized",
            403: "Forbidden",
            404: "Not Found",
            429: "Too Many Requests",
            500: "Internal Server Error",
        }
        content = _error_body(
            _labels.get(exc.status_code, "Error"),
            str(exc.detail),
            exc.status_code,
            str(request.url.path),
        )
    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(RequestValidationError)
async def validation_exc_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = exc.errors()
    messages = "; ".join(
        f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in errors
    )
    logger.warning("validation_error path=%s", request.url.path)
    return JSONResponse(
        status_code=400,
        content=_error_body("Bad Request", messages, 400, str(request.url.path)),
    )


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exc_handler(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    logger.warning(
        "rate_limit_exceeded ip=%s path=%s", _client_ip(request), request.url.path
    )
    return JSONResponse(
        status_code=429,
        content=_error_body(
            "Too Many Requests",
            "Rate limit exceeded. Please slow down.",
            429,
            str(request.url.path),
        ),
    )


@app.exception_handler(Exception)
async def generic_exc_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_exception path=%s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content=_error_body(
            "Internal Server Error",
            "An unexpected error occurred",
            500,
            str(request.url.path),
        ),
    )


# ===========================================================================
# PUBLIC ROUTES
# ===========================================================================
@app.get("/")
def root():
    return {"service": "ml-api", "status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "ok" if registry.loaded else "starting",
        "inference_backend": "databricks" if DATABRICKS_TOKEN else "local",
        "models_loaded": list(registry.models.keys()),
    }


@app.get("/ready")
def ready_check():
    if not registry.loaded:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "ready"}


# ===========================================================================
# PROTECTED ROUTES
# ===========================================================================
@app.get("/metadata")
def metadata(
    _auth: dict = Depends(role_required("admin", "user", "service")),
):
    """Return training metadata. Requires any authenticated role."""
    return registry.metadata


@app.post("/predict", response_model=PredictResponse)
@limiter.limit("60/minute", key_func=_jwt_subject)
def predict(
    request: Request,
    req: PredictRequest,
    _auth: dict = Depends(role_required("service", "admin")),
) -> PredictResponse:
    """Run ML inference. Restricted to service and admin roles."""
    return registry.predict(req)


# ===========================================================================
# CLI ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if cmd == "train":
        train_models()
    elif cmd == "serve":
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=PORT)
    else:
        print(__doc__)
        sys.exit(1)
