"""
intake_api.py — Microservice 1: Intake API (single-file edition).

A self-contained FastAPI service for user-facing data collection. All logic
that was previously split across app/main.py, app/schemas.py,
app/food_service.py, app/store.py and shared/feature_spec.py is collapsed into
this one file.

It manages user registration + demographic profiles, food search over the
USDA dataset, and dietary intake submissions. For each submitted food it looks
up the entry, scales the per-100g nutrient values by the submitted grams,
aggregates the totals, then forwards the combined profile + rolling daily
average nutrient totals to the ML API (Microservice 2) for disease risk
predictions. The rolling average uses up to the latest seven daily records and
also works before a full week has been entered.

--------------------------------------------------------------------------
DATA FILE (must sit next to this file, or set env var):
  food.csv  ........  USDA food composition dataset   (env: FOOD_CSV)
--------------------------------------------------------------------------

USAGE
  Run the API:
      python intake_api.py            # -> http://localhost:8001/docs

  Run with uvicorn directly:
      uvicorn intake_api:app --host 0.0.0.0 --port 8001

  The ML API location is taken from the ML_API_URL env var
  (default http://localhost:8002).

ENDPOINTS (public)
  GET  /                          — service banner
  GET  /health                    — liveness probe
  GET  /ready                     — readiness probe
  POST /v1/login                  — authenticate, receive JWT (5 req/min per IP)

ENDPOINTS (protected — requires Authorization: Bearer <token>)
  POST /users                     — register a user            [user, admin]
  GET  /users/{user_id}           — fetch a user profile       [user, admin, service]
  GET  /foods/search?q=           — search the USDA food dataset [user, admin, service]
  POST /users/{user_id}/intake    — submit a daily intake      [user, admin]
  GET  /users/{user_id}/intake    — intake history for a user  [user, admin, service]
"""

import datetime
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import httpx
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

from security import JWT_EXP_MINUTES, generate_token, peek_token, role_required

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("intake_api")

# ===========================================================================
# CONFIG / PATHS
# ===========================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
FOOD_CSV = os.environ.get("FOOD_CSV", os.path.join(HERE, "food.csv"))
PORT = int(os.environ.get("PORT", "8001"))

ML_API_URL = os.environ.get("ML_API_URL", "http://localhost:8002")
ML_TIMEOUT_SECONDS = float(os.environ.get("ML_API_TIMEOUT", "10"))
MAX_CONTENT_BYTES = 1 * 1024 * 1024  # 1 MB
PREDICTION_WINDOW_DAYS = 7

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8501"
).split(",")

# Demo user store — passwords sourced from env vars; fall back to obvious
# dev-only defaults. In production always set these via Kubernetes Secret.
DEMO_USERS: Dict[str, Dict[str, str]] = {
    "admin": {
        "password": os.environ.get("ADMIN_PASSWORD", "admin-password"),
        "role": "admin",
    },
    "user": {
        "password": os.environ.get("USER_PASSWORD", "user-password"),
        "role": "user",
    },
    "service": {
        "password": os.environ.get("SERVICE_PASSWORD", "service-password"),
        "role": "service",
    },
}

# Service JWT used for intake-api → ml-api calls.
# Lazily generated and refreshed automatically so it never expires mid-flight.
_SVC_TOKEN_LOCK = threading.Lock()
_svc_token: str = ""
_svc_token_exp: datetime.datetime = datetime.datetime.fromtimestamp(
    0, tz=datetime.timezone.utc
)


def _get_service_jwt() -> str:
    """Return a valid service JWT, regenerating it if it expires within 5 minutes."""
    global _svc_token, _svc_token_exp
    now = datetime.datetime.now(datetime.timezone.utc)
    if now >= _svc_token_exp - datetime.timedelta(minutes=5):
        with _SVC_TOKEN_LOCK:
            now = datetime.datetime.now(datetime.timezone.utc)
            if now >= _svc_token_exp - datetime.timedelta(minutes=5):
                exp_minutes = 24 * 60
                _svc_token = generate_token(
                    "intake-service", "service", exp_minutes=exp_minutes
                )
                _svc_token_exp = now + datetime.timedelta(minutes=exp_minutes)
                logger.info("service_jwt_refreshed expires_in_minutes=%d", exp_minutes)
    return _svc_token


# ===========================================================================
# FEATURE SPEC — the data contract shared with the ML API.
# ---------------------------------------------------------------------------
# NUTRIENT_FIELDS is the VERIFIED intersection of the two datasets. food.csv
# has no calorie/energy column and no caffeine column, so the NHANES fields
# `calories` and `caffeine_mg` cannot be sourced per-food. The honest common
# set is the 8 fields below.
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

# VERIFIED mapping: canonical field -> food.csv column name.
# Confirmed by inspecting the real food.csv header. Values are per-100g.
FOOD_COLUMN_MAP: Dict[str, str] = {
    "protein_g": "Data.Protein",
    "carbs_g": "Data.Carbohydrate",
    "sugar_g": "Data.Sugar Total",
    "fiber_g": "Data.Fiber",
    "total_fat_g": "Data.Fat.Total Lipid",
    "saturated_fat_g": "Data.Fat.Saturated Fat",
    "cholesterol_mg": "Data.Cholesterol",
    "sodium_mg": "Data.Major Minerals.Sodium",
}


# ===========================================================================
# SCHEMAS (Pydantic request/response models)
# ===========================================================================
class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


class LoginResponse(BaseModel):
    token: str
    role: str
    expires_in_minutes: int


class UserProfile(BaseModel):
    """Demographic profile collected at registration.

    Codings follow the NHANES dataset so values flow straight to the ML API.
    """

    name: str = Field(..., min_length=1, description="Display name")
    age: float = Field(..., ge=0, le=120)
    gender: int = Field(..., ge=1, le=2, description="1=male, 2=female (NHANES)")
    race_ethnicity: int = Field(..., ge=1, le=5, description="NHANES coding 1-5")
    education_level: int = Field(..., ge=1, le=5, description="NHANES coding 1-5")
    weight_kg: float = Field(..., gt=0, le=400)
    height_cm: float = Field(..., gt=0, le=260)


class UserResponse(BaseModel):
    user_id: str
    profile: UserProfile


class FoodItemInput(BaseModel):
    food_name: str = Field(..., min_length=1, description="Food name to look up")
    grams: float = Field(..., gt=0, le=10000, description="Submitted weight in grams")


class IntakeSubmission(BaseModel):
    items: List[FoodItemInput] = Field(..., min_length=1)
    intake_date: Optional[datetime.date] = Field(
        None,
        description="Date this daily intake belongs to. Defaults to today's UTC date.",
    )


class ResolvedItem(BaseModel):
    submitted_name: str
    matched_description: str
    grams: float
    scaled_nutrients: Dict[str, float]


class IntakeRecord(BaseModel):
    """A single aggregated daily intake record + the ML prediction for it.

    `nutrient_totals` is the submitted day's total. The model prediction is
    based on `weekly_average_nutrient_totals`: the average of this day plus the
    previous available daily records, capped at seven days.
    """

    record_id: str
    user_id: str
    timestamp: str
    intake_date: datetime.date
    nutrient_totals: Dict[str, float]
    weekly_average_nutrient_totals: Dict[str, float]
    prediction_window_days: int
    resolved_items: List[ResolvedItem]
    unmatched_items: List[str]
    prediction: Optional[Dict] = Field(
        None, description="Response from ML API, or null if ML API unreachable"
    )
    prediction_error: Optional[str] = None


# ===========================================================================
# IN-MEMORY STORE
# ---------------------------------------------------------------------------
# Plain in-memory store. Persistent structured storage belongs to the
# Databricks/Snowflake component (out of scope for the API deliverable).
# Swapping in a real database means replacing this one class.
# ===========================================================================
class InMemoryStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._users: Dict[str, dict] = {}
        self._intakes: Dict[str, List[dict]] = {}
        self._username_to_user_id: Dict[str, str] = {}

    def add_user(self, user_id: str, record: dict) -> None:
        with self._lock:
            self._users[user_id] = record

    def get_user(self, user_id: str) -> Optional[dict]:
        return self._users.get(user_id)

    def user_exists(self, user_id: str) -> bool:
        return user_id in self._users

    def map_username(self, username: str, user_id: str) -> None:
        with self._lock:
            self._username_to_user_id[username] = user_id

    def get_user_id_for_username(self, username: str) -> Optional[str]:
        return self._username_to_user_id.get(username)

    def add_intake(self, user_id: str, record: dict) -> None:
        with self._lock:
            self._intakes.setdefault(user_id, []).append(record)

    def get_intakes(self, user_id: str) -> List[dict]:
        return list(self._intakes.get(user_id, []))

    def patch_intake_prediction(
        self, user_id: str, date_str: str, prediction: dict
    ) -> bool:
        """Overwrite the prediction on the record matching intake_date == date_str.

        Returns True if a record was found and updated, False otherwise.
        Matches on the most-recently-submitted record for that date if duplicates exist.
        """
        with self._lock:
            records = self._intakes.get(user_id, [])
            matched = [
                r for r in records if str(r.get("intake_date", ""))[:10] == date_str
            ]
            if not matched:
                return False
            target = matched[-1]
            target["prediction"] = prediction
            target["prediction_error"] = None
            return True


store = InMemoryStore()


def _assert_resource_access(auth: dict, user_id: str) -> None:
    """Raise 403 if a non-admin/service token tries to access another user's record."""
    if auth.get("role") in ("admin", "service"):
        return
    token_user_id = store.get_user_id_for_username(auth.get("sub", ""))
    if token_user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Forbidden",
                "message": "You can only access your own records",
                "status_code": 403,
            },
        )


# ===========================================================================
# FOOD SERVICE — USDA dataset lookup and nutrient aggregation
# ===========================================================================
class FoodService:
    """In-memory USDA food table with search and aggregation."""

    def __init__(self) -> None:
        self.df: pd.DataFrame = pd.DataFrame()
        self.loaded = False

    def load(self) -> None:
        if not os.path.exists(FOOD_CSV):
            raise FileNotFoundError(f"food.csv not found at {FOOD_CSV}")
        df = pd.read_csv(FOOD_CSV)

        missing = [c for c in FOOD_COLUMN_MAP.values() if c not in df.columns]
        if missing:
            raise ValueError(f"food.csv is missing expected columns: {missing}")

        keep = ["Category", "Description"] + list(FOOD_COLUMN_MAP.values())
        df = df[keep].copy()
        for col in FOOD_COLUMN_MAP.values():
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        df["_search"] = df["Description"].str.lower()
        self.df = df.reset_index(drop=True)
        self.loaded = True
        logger.info("Loaded food dataset: %d foods", len(self.df))

    def search_foods(self, query: str, limit: int = 20) -> List[Dict]:
        """Case-insensitive substring search over food Description."""
        if not query.strip():
            return []
        q = query.strip().lower()
        hits = self.df[self.df["_search"].str.contains(q, regex=False, na=False)]
        results = []
        for _, row in hits.head(limit).iterrows():
            results.append(
                {
                    "description": row["Description"],
                    "category": row["Category"],
                    "per_100g": {f: float(row[c]) for f, c in FOOD_COLUMN_MAP.items()},
                }
            )
        return results

    def get_food(self, description: str) -> Optional[Dict]:
        """Exact (case-insensitive) lookup by Description. None if not found."""
        match = self.df[self.df["_search"] == description.strip().lower()]
        if match.empty:
            return None
        row = match.iloc[0]
        return {
            "description": row["Description"],
            "category": row["Category"],
            "per_100g": {f: float(row[c]) for f, c in FOOD_COLUMN_MAP.items()},
        }

    def aggregate_intake(self, items: List[Dict]) -> Dict:
        """Scale each food's per-100g values by submitted grams and sum.

        Scaling: total_for_item = per_100g_value * (grams / 100).
        Returns {"totals", "resolved", "unmatched"}.
        """
        totals = {field: 0.0 for field in NUTRIENT_FIELDS}
        resolved: List[Dict] = []
        unmatched: List[str] = []

        for item in items:
            name = item["food_name"]
            grams = float(item["grams"])
            food = self.get_food(name)
            if food is None:
                # Fall back to best substring match so a near-name still works.
                search_hits = self.search_foods(name, limit=1)
                if not search_hits:
                    unmatched.append(name)
                    continue
                food = search_hits[0]

            factor = grams / 100.0
            scaled = {}
            for field in NUTRIENT_FIELDS:
                value = food["per_100g"].get(field, 0.0) * factor
                scaled[field] = round(value, 4)
                totals[field] += value

            resolved.append(
                {
                    "submitted_name": name,
                    "matched_description": food["description"],
                    "grams": grams,
                    "scaled_nutrients": scaled,
                }
            )

        totals = {k: round(v, 4) for k, v in totals.items()}
        return {"totals": totals, "resolved": resolved, "unmatched": unmatched}


food_service = FoodService()


# ===========================================================================
# HELPERS
# ===========================================================================
def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _today_utc() -> datetime.date:
    return datetime.datetime.now(datetime.timezone.utc).date()


def _error_body(error: str, message: str, status_code: int, path: str) -> dict:
    return {
        "error": error,
        "message": message,
        "status_code": status_code,
        "path": path,
        "timestamp": _now_iso(),
    }


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


limiter = Limiter(key_func=_real_client_ip)


# ===========================================================================
# FASTAPI APP — lifespan, middleware, exception handlers
# ===========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    food_service.load()
    logger.info("Intake API ready. ML_API_URL=%s", ML_API_URL)
    yield


app = FastAPI(
    title="Dietary Intake API",
    description="Microservice 1 — user profiles, food search, intake aggregation.",
    version="2.0.0",
    lifespan=lifespan,
    root_path="/api",
)
app.state.limiter = limiter


# ----- security headers + request logging (inner middleware) -----
@app.middleware("http")
async def log_and_secure(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_CONTENT_BYTES:
        return JSONResponse(
            status_code=413,
            content={
                "error": "Payload Too Large",
                "message": f"Request body exceeds {MAX_CONTENT_BYTES // 1024} KB",
                "status_code": 413,
                "path": str(request.url.path),
                "timestamp": _now_iso(),
            },
        )

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


# ----- CORS (outer middleware — runs first on requests) -----
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


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
            405: "Method Not Allowed",
            422: "Unprocessable Entity",
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
    logger.warning("validation_error path=%s errors=%s", request.url.path, errors)
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
    return {"service": "intake-api", "status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "ok" if food_service.loaded else "starting",
        "foods_loaded": len(food_service.df),
        "ml_api_url": ML_API_URL,
    }


@app.get("/ready")
def ready():
    if not food_service.loaded:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready"}


@app.post("/v1/login", response_model=LoginResponse)
@limiter.limit("5/minute")
async def login(request: Request, body: LoginRequest) -> LoginResponse:
    """Authenticate with username + password and receive a signed JWT.

    Rate-limited to 5 attempts per minute per IP address.
    """
    user = DEMO_USERS.get(body.username)
    if user is None or user["password"] != body.password:
        logger.warning(
            "auth_failure username=%s ip=%s", body.username, _client_ip(request)
        )
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Forbidden",
                "message": "Invalid credentials",
                "status_code": 403,
            },
        )
    token = generate_token(body.username, user["role"])
    logger.info(
        "login_success username=%s role=%s ip=%s",
        body.username,
        user["role"],
        _client_ip(request),
    )
    return LoginResponse(
        token=token, role=user["role"], expires_in_minutes=JWT_EXP_MINUTES
    )


# ===========================================================================
# PROTECTED ROUTES — user management
# ===========================================================================
@app.post("/users", response_model=UserResponse, status_code=201)
@limiter.limit("20/minute")
def register_user(
    request: Request,
    profile: UserProfile,
    _auth: dict = Depends(role_required("admin", "user")),
) -> UserResponse:
    """Register a user with a demographic profile."""
    user_id = str(uuid.uuid4())
    record = {"user_id": user_id, "profile": profile.model_dump()}
    store.add_user(user_id, record)
    store.map_username(_auth.get("sub", ""), user_id)
    logger.info("user_registered id=%s by=%s", user_id, _auth.get("sub"))
    return UserResponse(**record)


@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(
    user_id: str,
    _auth: dict = Depends(role_required("admin", "user", "service")),
) -> UserResponse:
    _assert_resource_access(_auth, user_id)
    record = store.get_user(user_id)
    if record is None:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(**record)


# ===========================================================================
# PROTECTED ROUTES — food search
# ===========================================================================
@app.get("/foods/search")
def search_foods(
    q: str = Query(..., min_length=1, description="Food name search query"),
    limit: int = Query(20, ge=1, le=100),
    _auth: dict = Depends(role_required("admin", "user", "service")),
):
    """Search the USDA food composition dataset by name."""
    results = food_service.search_foods(q, limit=limit)
    return {"query": q, "count": len(results), "results": results}


# ===========================================================================
# PROTECTED ROUTES — dietary intake
# ===========================================================================
def _call_ml_api(payload: dict) -> tuple[Optional[dict], Optional[str]]:
    """POST the aggregated payload to the ML API with a service-level JWT.

    Returns (prediction, error). Exactly one is non-None. The intake record is
    still saved if the ML API is unreachable, so no user data is lost.
    """
    try:
        resp = httpx.post(
            f"{ML_API_URL}/predict",
            json=payload,
            timeout=ML_TIMEOUT_SECONDS,
            headers={"Authorization": f"Bearer {_get_service_jwt()}"},
        )
        resp.raise_for_status()
        return resp.json(), None
    except httpx.HTTPStatusError as e:
        msg = f"ML API returned {e.response.status_code}: {e.response.text[:200]}"
        logger.warning("ml_api_error %s", msg)
        return None, msg
    except httpx.RequestError as e:
        msg = f"ML API unreachable: {e}"
        logger.warning("ml_api_error %s", msg)
        return None, msg


def _record_sort_key(record: dict) -> tuple[str, str]:
    """Order intake records by their intake day, then submission timestamp."""
    intake_date = record.get("intake_date")
    if intake_date is None:
        intake_date = str(record.get("timestamp", ""))[:10]
    return str(intake_date), str(record.get("timestamp", ""))


def _prediction_window(
    user_id: str,
    current_totals: Dict[str, float],
    intake_date: datetime.date,
    timestamp: str,
) -> tuple[Dict[str, float], int]:
    """Average the latest available daily records, capped at seven."""
    current_record = {
        "timestamp": timestamp,
        "intake_date": intake_date.isoformat(),
        "nutrient_totals": current_totals,
    }
    records = sorted(
        [*store.get_intakes(user_id), current_record], key=_record_sort_key
    )
    window_records = records[-PREDICTION_WINDOW_DAYS:]
    window_totals = [
        {
            field: float(record.get("nutrient_totals", {}).get(field, 0.0))
            for field in NUTRIENT_FIELDS
        }
        for record in window_records
    ]

    window_days = len(window_totals)
    averages = {
        field: round(sum(day[field] for day in window_totals) / window_days, 4)
        for field in NUTRIENT_FIELDS
    }
    return averages, window_days


@app.post("/users/{user_id}/intake", response_model=IntakeRecord)
@limiter.limit("30/minute")
def submit_intake(
    request: Request,
    user_id: str,
    submission: IntakeSubmission,
    _auth: dict = Depends(role_required("admin", "user")),
) -> IntakeRecord:
    """Submit a daily dietary intake.

    Steps: (1) validate user, (2) look up each food and scale per-100g values
    by grams then aggregate, (3) average this day with the latest available
    daily records up to a seven-day window, (4) forward profile + averages to
    the ML API, (5) persist and return the combined record.
    """
    _assert_resource_access(_auth, user_id)
    user = store.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    items = [item.model_dump() for item in submission.items]
    aggregated = food_service.aggregate_intake(items)

    if not aggregated["resolved"]:
        raise HTTPException(
            status_code=422,
            detail=f"No submitted foods matched the dataset: {aggregated['unmatched']}",
        )

    intake_date = submission.intake_date or _today_utc()
    if intake_date > _today_utc():
        raise HTTPException(
            status_code=422, detail="intake_date cannot be in the future"
        )
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    weekly_average_totals, prediction_window_days = _prediction_window(
        user_id, aggregated["totals"], intake_date, timestamp
    )

    profile = user["profile"]
    ml_payload = {
        "age": profile["age"],
        "gender": profile["gender"],
        "race_ethnicity": profile["race_ethnicity"],
        "education_level": profile["education_level"],
        "weight_kg": profile["weight_kg"],
        "height_cm": profile["height_cm"],
        **weekly_average_totals,
    }
    prediction, error = _call_ml_api(ml_payload)

    record = IntakeRecord(
        record_id=str(uuid.uuid4()),
        user_id=user_id,
        timestamp=timestamp,
        intake_date=intake_date,
        nutrient_totals=aggregated["totals"],
        weekly_average_nutrient_totals=weekly_average_totals,
        prediction_window_days=prediction_window_days,
        resolved_items=aggregated["resolved"],
        unmatched_items=aggregated["unmatched"],
        prediction=prediction,
        prediction_error=error,
    )
    store.add_intake(user_id, record.model_dump())
    return record


@app.patch("/users/{user_id}/intake/{date}")
def patch_intake_prediction(
    request: Request,
    user_id: str,
    date: str,
    body: Dict,
    _auth: dict = Depends(role_required("admin", "user", "service")),
) -> Dict:
    """Overwrite the stored ML prediction for one day's intake record.

    Used by the demo seed to inject hardcoded predictions so the dashboard
    charts reflect the intended risk level without needing the ML API to return
    specific values.
    """
    _assert_resource_access(_auth, user_id)
    if not store.user_exists(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    updated = store.patch_intake_prediction(user_id, date, body)
    if not updated:
        raise HTTPException(
            status_code=404, detail=f"No intake record found for date {date}"
        )
    return {"updated": True, "date": date}


@app.get("/users/{user_id}/intake")
def intake_history(
    user_id: str,
    _auth: dict = Depends(role_required("admin", "user", "service")),
):
    """Return all stored intake records for a user, newest last."""
    _assert_resource_access(_auth, user_id)
    if not store.user_exists(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    history = store.get_intakes(user_id)
    return {"user_id": user_id, "count": len(history), "records": history}


# ===========================================================================
# CLI ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
