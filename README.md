# Nutrition Support Platform

Log what you ate this week, get back a health risk assessment explained in plain
English.

The platform resolves foods against a USDA composition database, aggregates the
nutrients, scores the result against models trained on real CDC survey data, and
has an LLM write up what the numbers mean. Every figure in that write-up comes
from a deterministic tool, never from the model.

> Informational only. This is not a medical diagnosis, and the disclaimer is
> appended in code rather than left to the language model to remember.

### At a glance

| | |
|---|---|
| **Backend** | Two FastAPI microservices, JWT auth, RBAC, rate limiting |
| **ML** | 4 scikit-learn classifiers trained locally, plus a PySpark pipeline served from Databricks |
| **AI** | Two-agent LLM workflow built on OpenAI tool calling |
| **Frontend** | Streamlit client with Plotly dashboards |
| **Infra** | Docker, Kubernetes/AKS, NetworkPolicy, GitHub Actions CI |
| **Tests** | 35 tests covering the Intake API, with outbound calls stubbed |
| **Data** | NHANES 2013-2014 (2,803 respondents), USDA food data (7,083 foods) |

---

## What I built

**Nutrient resolution and aggregation.** Foods are matched against a 7,083-row
USDA dataset using exact match, then case-insensitive, then substring fallback.
Per-100g values scale by the grams submitted and sum into daily totals across 8
tracked nutrients. Unmatched items come back in the response rather than getting
silently dropped.

**A rolling 7-day risk window.** Predictions average up to the seven most recent
daily records. It works correctly before a full week of data exists. Intake can
be backfilled to a past date, and future dates are rejected.

**Two inference paths.** Four local RandomForest classifiers, one per condition,
and an optional Spark ML pipeline hosted on Databricks. The service picks
between them at runtime based on whether `DATABRICKS_TOKEN` is set.

**A two-agent LLM workflow.** An analyzer agent computes nutritional gaps
against FDA reference values. A reporter agent scores the risk model and writes
the summary. An orchestrator chains them and captures what the tools actually
returned, so the output is verifiable independently of the prose.

**Security built into the services.** JWT authentication with automatic token
refresh, three-role RBAC, resource-based authorization, and per-identity rate
limiting. Details in the [Security](#security) section.

**Graceful degradation.** If the ML API is down or returns a 5xx, the user's
meal is still saved. The response carries a `prediction_error` field instead of
failing the write.

---

## The problem

Diet is one of the most controllable factors in chronic disease, but there is no
simple bridge between "what I ate this week" and a meaningful health signal.

Consumer calorie trackers report totals. They do not relate those totals to
clinical risk markers or to a person's demographic context. Interpreting raw
intake logs against reference values is tedious and needs nutritional expertise
most people do not have.

This platform closes that loop end to end.

---

## Architecture

```
                   +--------------------------+
                   |   Streamlit frontend     |  :8501
                   |  profile / log / charts  |
                   +------------+-------------+
                                | JWT
                                v
+---------------------------------------------------------------+
|  Intake API  (FastAPI)                                   :8001 |
|  auth, profiles, food search, intake aggregation               |
|    USDA food.csv (7,083 foods)                                 |
|    rolling 7-day nutrient average                              |
+----------+--------------------------------+-------------------+
           | service JWT                     |
           v                                 v
+------------------------+     +--------------------------------+
|  ML API  (FastAPI)     |     |  Agentic workflow              |
|                  :8002 |     |                                |
|  4 x RandomForest      |     |  Agent 1: Nutritional Analyzer |
|  trained from NHANES   |     |    computes gaps vs FDA values |
|                        |     |             | handoff          |
|  if DATABRICKS_TOKEN:  |     |             v                  |
|    delegate ---------------->|  Agent 2: Health Risk Reporter |
|    to hosted endpoint  |     |    tool: predict_health_risk   |
+------------------------+     +----------------+---------------+
                                                |
                  +-----------------------------v--------------+
                  |  Databricks / MLflow serving endpoint      |
                  |  Spark ML pipeline, 106 features           |
                  +--------------------------------------------+
```

The ML API is never exposed externally. In Kubernetes both services are
`ClusterIP`, only the Intake API is reachable through the Ingress, and a
`NetworkPolicy` restricts traffic on the ML API's port 8002 to pods labelled
`app: intake-api`.

---

## Technology stack

| Layer | Technology |
|---|---|
| API services | FastAPI, Pydantic v2, Uvicorn |
| Auth and limits | PyJWT (HS256), SlowAPI |
| Local ML | scikit-learn `RandomForestClassifier`, joblib |
| Distributed ML | PySpark ML, MLflow Model Registry, Databricks Unity Catalog |
| LLM agents | OpenAI tool-calling API (default `gpt-4o`) |
| Frontend | Streamlit, Plotly |
| Data | pandas, NumPy |
| Containers | Docker, with per-service build contexts |
| Orchestration | Kubernetes/AKS: Deployments, Services, Ingress, NetworkPolicy, RBAC |
| CI | GitHub Actions: Ruff lint and format, Docker build matrix |
| Security testing | OWASP ZAP baseline scan |
| Tests | pytest, respx, FastAPI `TestClient` |

---

## Data and ML

### Datasets

Both are US Government public-domain works, committed so the project runs
without an external download.

`intake_api/food.csv` holds USDA food composition data: 7,083 foods across 38
nutrient columns, per 100g.

`ml_api/nhanes.csv` holds CDC NHANES 2013-2014 data: 2,803 de-identified
respondents across 116 columns, joined from the demographic, dietary,
examination, laboratory, and medication survey files.

### Local models

Four independent binary classifiers (hypertension, hypercholesterolemia, type 2
diabetes, GERD) trained on 15 features covering demographics, body measures, and
the 8 aggregated nutrients. An 80/20 stratified split, `n_estimators=200`,
`random_state=42`. Training runs from the committed CSV via `python ml_api.py
train`, and the Dockerfile executes it at build time so containers start with
models already loaded.

Held-out test metrics, recorded in `ml_api/models/metadata.json`:

| Condition | Test AUC | Test accuracy | Positive cases | Prevalence |
|---|---|---|---|---|
| Hypertension | 0.887 | 0.813 | 434 | 15.5% |
| Hypercholesterolemia | 0.904 | 0.825 | 292 | 10.4% |
| Type 2 diabetes | 0.864 | 0.866 | 173 | 6.2% |
| GERD | 0.796 | 0.938 | 113 | 4.0% |

These are imbalanced targets, so accuracy is the weaker signal here. GERD's
0.938 accuracy sits against a 4.0% prevalence, which a majority-class predictor
would nearly match. AUC is the number to read.

### Distributed pipeline

`Data Ingestion.ipynb` is a Databricks notebook that ingests the five NHANES
source tables into Unity Catalog, drops laboratory columns exceeding 40% nulls,
aggregates medications per respondent, joins on `SEQN`, and fits a Spark ML
pipeline of median `Imputer`, `VectorAssembler`, then `RandomForestClassifier`.
The model is registered through MLflow and exposed as a REST serving endpoint.

Its label, `high_health_risk_label`, derives from BMI at or above 30, or blood
pressure at or above 130/80. BMI and the blood-pressure columns are therefore
excluded from the feature set to prevent label leakage. That is why the hosted
model uses 106 features where the local models use 15.

---

## API design

Both services expose `/health`, `/ready`, and a service banner at `/`, all
unauthenticated so Kubernetes can probe them. Everything else requires
`Authorization: Bearer <token>`.

### Intake API, port 8001

| Method | Path | Roles | Purpose |
|---|---|---|---|
| `POST` | `/v1/login` | public | Authenticate, receive a JWT. 5 req/min per IP |
| `POST` | `/users` | user, admin | Register a user and demographic profile |
| `GET` | `/users/{user_id}` | user, admin, service | Fetch a profile |
| `GET` | `/foods/search?q=` | user, admin, service | Search the USDA dataset |
| `POST` | `/users/{user_id}/intake` | user, admin | Submit a daily intake |
| `GET` | `/users/{user_id}/intake` | user, admin, service | Intake history |

Intake submission takes an optional `intake_date` for backfilling a missed day,
defaulting to today's UTC date.

### ML API, port 8002

| Method | Path | Roles | Purpose |
|---|---|---|---|
| `GET` | `/metadata` | user, admin, service | Training metadata and metrics |
| `POST` | `/predict` | service, admin | Run inference. 60 req/min per JWT subject |

`/predict` returns `overall_health_risk` (bool), `overall_probability` (float),
and a `disease_flags` object with the per-condition result.

Note the role split: `/predict` is closed to the `user` role entirely. Only the
Intake API's service identity and admins reach inference directly.

---

## Security

The layers are independent, so no single failure exposes the system.

**JWT authentication.** HS256, 60-minute expiry, signing key from
`JWT_SECRET_KEY`. The Intake API holds a service token for its outbound calls to
the ML API and regenerates it automatically within 5 minutes of expiry, so
inference never fails on a stale token mid-request.

**Role-based access control.** Three roles (`user`, `admin`, `service`) enforced
by a `role_required(*roles)` FastAPI dependency applied per route.

**Resource-based authorization.** A `user` token reads and modifies only its own
records. Requesting another user's data returns 403. `admin` and `service`
bypass this deliberately, because the agentic workflow needs to read arbitrary
user IDs. That is why the workflow authenticates as `service`, not `user`.

**Rate limiting.** SlowAPI, tiered per endpoint: 5/min for login, 20/min and
30/min for data routes. The ML API keys its 60/min limit on the JWT subject
rather than the client IP, because every in-cluster call arrives from the same
pod IP and would otherwise share a single bucket.

**CORS.** Explicit origin allow-list, restricted to `GET` and `POST` and to the
`Authorization` and `Content-Type` headers. No wildcards.

**Network isolation.** A Kubernetes `NetworkPolicy` permits traffic to the ML
API's port 8002 only from pods labelled `app: intake-api`.

**Kubernetes RBAC.** A read-only `Role` scoped to the `diet-risk` namespace,
bound to a dedicated ServiceAccount. The default ServiceAccount is explicitly
not bound.

**Secret management.** No secrets in source. `k8s/secrets.yaml` is a documented
template, and real values are created with `kubectl create secret` and injected
as environment variables.

**Automated scanning.** An OWASP ZAP baseline scan runs in GitHub Actions,
configured by `zap.yaml`.

Manual verification steps for every control above are written up in
[`security/security_testing.md`](security/security_testing.md).

### Known limits

The user store is an in-memory dictionary with environment-sourced passwords,
suitable for demonstration rather than production. Passwords are compared
directly rather than hashed, and intake data does not survive a restart.
Replacing the store with a database and a proper password hash is the first item
in the roadmap below.

---

## Engineering decisions

**Single-file services.** Each microservice is one Python module rather than a
package. At this size, the indirection of `app/main.py` plus `app/schemas.py`
plus `app/store.py` cost more than it bought. One file per service keeps the
whole request path readable top to bottom.

**Per-service Docker build contexts.** Each service owns its `Dockerfile` and
`.dockerignore`, so building the ML image never ships the Intake API's 2 MB food
dataset to the daemon, and the two services keep independent dependency sets.

**Models baked in at image build.** The ML Dockerfile runs `python ml_api.py
train` as a build step. Containers start fast and are self-contained, at the
cost of a longer build. That is the right trade for a service meant to scale
horizontally.

**Two Ingress objects instead of one.** The NGINX `rewrite-target` annotation
applies to every path in an Ingress, not only the regex path it was written for.
With a single object, `/docs` had no capture group, `$2` resolved to an empty
string, and Swagger UI silently served the root handler instead. Splitting
`/docs` and `/openapi.json` (no rewrite) from `/api/*` (with rewrite) fixes it.
See [`k8s/ingress.yaml`](k8s/ingress.yaml).

**Rate limiting keyed on JWT subject for internal traffic.** Covered above.
IP-keyed limits are meaningless behind a single calling pod.

**Deterministic tools, generative prose.** The agents' numbers never come from
the language model. Nutrient math and the risk prediction run in tool functions.
The model decides when to call them and writes the explanation around the
results. A `WorkflowContext` captures what the tools actually computed, so
`--json` output can be checked independently of the prose.

**Null-filled feature record with server-side imputation.** The platform can
source only a fraction of the hosted model's 106 columns. The rest are sent as
explicit `null` and filled by the pipeline's own median `Imputer` at inference
time, keeping imputation policy with the model instead of duplicating it in the
client.

**Graceful ML degradation.** A failed prediction must not lose a user's logged
meal. Intake writes commit first, and prediction failures surface as a
`prediction_error` field on an otherwise successful response.

---

## Project structure

```
.
├── intake_api/              Microservice 1: user-facing data collection
│   ├── intake_api.py        all service logic in one file
│   ├── security.py          JWT issue/verify, RBAC dependencies
│   ├── food.csv             USDA food composition data (7,083 foods)
│   ├── Dockerfile
│   └── tests/               35 tests, unit and API integration
├── ml_api/                  Microservice 2: inference
│   ├── ml_api.py            training and serving in one file
│   ├── security.py          identical to intake_api/security.py
│   ├── nhanes.csv           NHANES 2013-2014 training data (2,803 rows)
│   ├── models/              trained .joblib models and metadata.json
│   └── Dockerfile
├── agents/                  Two-agent LLM workflow
│   ├── llm.py               generic OpenAI tool-calling runtime
│   ├── clients.py           Intake API and Databricks REST clients
│   ├── nutrition_agent.py   Agent 1 and its tools
│   ├── risk_agent.py        Agent 2 and the 106-column feature record
│   ├── feature_schema.py    the model's exact input signature
│   └── workflow.py          orchestrator and CLI
├── frontend/                Streamlit client
│   ├── app.py               four-tab UI
│   ├── intake_client.py     httpx client over the Intake API
│   └── workflow_runner.py   bridges the UI to the agents package
├── k8s/                     AKS manifests
│   ├── namespace.yaml       intake-api.yaml     ingress.yaml
│   ├── ml-api.yaml          network-policy.yaml secrets.yaml (template)
│   └── role.yaml            role-binding.yaml
├── security/                ZAP scan config and security testing guide
├── Data Ingestion.ipynb     Databricks ETL and Spark ML training notebook
└── .github/workflows/       CI and scheduled security scan
```

Each service resolves its dataset path relative to its own file, so both run
from their own folder with no path configuration.

---

## Running it locally

Requires Python 3.12 or newer.

```bash
git clone <your-repo-url>
cd ai-nutrition-support-platform

python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1

cp .env.example .env             # then fill in any values you need
```

Only the agentic workflow needs external credentials. The APIs, the models, and
the dashboard all run offline from the committed datasets.

```bash
# Terminal 1: ML API. Auto-trains on first run if models/ is absent.
cd ml_api && python ml_api.py                     # -> localhost:8002/docs

# Terminal 2: Intake API
cd intake_api && ML_API_URL=http://localhost:8002 python intake_api.py
                                                  # -> localhost:8001/docs

# Terminal 3: Streamlit frontend
pip install -r frontend/requirements.txt
streamlit run frontend/app.py                     # -> localhost:8501
```

`python ml_api.py train` trains without serving. `python ml_api.py serve` serves
and auto-trains if `models/` is missing.

### With Docker

```bash
docker build -t ml-api:latest     ./ml_api
docker build -t intake-api:latest ./intake_api

docker run --rm -p 8002:8002 --name ml-api ml-api:latest
docker run --rm -p 8001:8001 \
  -e ML_API_URL=http://host.docker.internal:8002 \
  --name intake-api intake-api:latest
```

### Configuration

| Variable | Default | Used by |
|---|---|---|
| `ML_API_URL` | `http://localhost:8002` | Intake API to ML API |
| `INTAKE_API_URL` | `http://localhost:8001` | frontend, agents |
| `JWT_SECRET_KEY` | dev fallback | both services. Set this in production |
| `JWT_EXP_MINUTES` | `60` | both services |
| `ALLOWED_ORIGINS` | `localhost:3000,localhost:8501` | Intake API CORS |
| `ADMIN_PASSWORD`, `USER_PASSWORD`, `SERVICE_PASSWORD` | dev fallbacks | Intake API user store |
| `DEMO_PASSWORD` | `user-password` | password hint in the Streamlit sidebar |
| `OPENAI_API_KEY` | none | required for the agents |
| `OPENAI_MODEL` | `gpt-4o` | the agents |
| `DATABRICKS_TOKEN` | none | required for hosted inference. Also switches `/predict` to delegate |
| `DATABRICKS_ENDPOINT_URL` | placeholder | the MLflow `/invocations` URL |

---

## Example usage

```bash
# 1. Authenticate
TOKEN=$(curl -s -X POST localhost:8001/v1/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"user","password":"user-password"}' | jq -r .token)

# 2. Register a profile
curl -s -X POST localhost:8001/users \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"age":54,"gender":1,"race_ethnicity":3,"education_level":3,
       "weight_kg":89.5,"height_cm":176.8,"medication_count":0}'

# 3. Search the food dataset
curl -s "localhost:8001/foods/search?q=apple" -H "Authorization: Bearer $TOKEN"

# 4. Log a day's intake. Returns nutrient totals plus the risk prediction.
curl -s -X POST localhost:8001/users/$USER_ID/intake \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"items":[{"food_name":"Apple raw","grams":150},
                {"food_name":"Beef bacon","grams":80}]}'
```

Run the agentic workflow over a user's history:

```bash
pip install -r agents/requirements.txt
cd agents
python workflow.py --user-id <USER_ID>            # prose report
python workflow.py --user-id <USER_ID> --json     # plus structured tool output
```

Every setting has a CLI override (`--model`, `--intake-url`, `--intake-token`
and others). Run `python workflow.py -h` for the full list.

---

## Testing

```bash
pip install -r intake_api/requirements-dev.txt
pytest intake_api/tests -v
```

35 tests across two files.

`test_food_service.py` holds 13 pure logic tests with no HTTP: nutrient scaling
linearity, field-by-field summation, unmatched-item collection, case-insensitive
and substring matching, empty-input handling, and the canonical field set.

`test_api.py` holds 22 tests covering the full HTTP surface through
`TestClient`, with outbound ML calls stubbed by `respx` so no live ML service is
needed. It covers registration, validation (422 on out-of-range gender, age over
120, blank query, empty items), 404s on unknown users, the rolling 7-day
average, backfilled and rejected future dates, and both ML-failure degradation
paths.

The fixtures use a two-row CSV with known per-100g values and reload the module
between tests to reset global state, so assertions are exact rather than
approximate.

---

## Deployment

Manifests in [`k8s/`](k8s/) target AKS with the managed NGINX ingress
(application routing add-on).

```bash
# 1. Build and push to your container registry
docker build -t <ACR_NAME>/ml-api:latest     ./ml_api
docker build -t <ACR_NAME>/intake-api:latest ./intake_api
docker push <ACR_NAME>/ml-api:latest
docker push <ACR_NAME>/intake-api:latest

# 2. Create the secret. Never commit real values.
kubectl create secret generic api-secret --namespace diet-risk \
  --from-literal=JWT_SECRET_KEY="$(python -c 'import secrets;print(secrets.token_hex(32))')" \
  --from-literal=ADMIN_PASSWORD="<strong-password>" \
  --from-literal=USER_PASSWORD="<strong-password>" \
  --from-literal=SERVICE_PASSWORD="<strong-password>" \
  --from-literal=DATABRICKS_TOKEN="<databricks-pat>"

# 3. Set your registry in k8s/ml-api.yaml and k8s/intake-api.yaml, then apply
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/role.yaml -f k8s/role-binding.yaml
kubectl apply -f k8s/network-policy.yaml
kubectl apply -f k8s/ml-api.yaml
kubectl apply -f k8s/intake-api.yaml
kubectl apply -f k8s/ingress.yaml
```

Enable the ingress add-on once per cluster with `az aks approuting enable
--resource-group <RG> --name <CLUSTER>`, or switch `ingressClassName` to `nginx`
for the community chart.

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs Ruff lint and
format checks and builds both Docker images on every push and pull request. The
matrix has fail-fast disabled, so one service failing still reports the other.

---

## Limitations and next steps

Known gaps, in the order worth addressing:

1. **Persistence.** Users and intake records live in an in-memory dict and are
   lost on restart. A Postgres-backed store with SQLAlchemy is the next step.
2. **Password hashing.** The demo user store compares plaintext. Argon2 or
   bcrypt with per-user salts should replace it alongside the database.
3. **Test coverage for the ML API and agents.** Only the Intake API has tests.
   The inference path and the agent tool-calling loop deserve the same
   treatment, and CI should lint all four packages rather than just the two API
   folders.
4. **Nutrient coverage.** The USDA file has no calorie or caffeine column, so
   aggregation covers the 8 nutrients common to both datasets rather than 10.
5. **Feature coverage for the hosted model.** The platform sources a handful of
   106 columns and the rest rely on median imputation. Collecting lab values or
   waist measurements would sharpen the hosted prediction.
6. **Integer-null schema caveat.** MLflow warned at logging time that the
   inferred signature contains integer columns, which cannot represent nulls.
   The columns actually filled are always real integers, but re-logging the
   signature with those columns as `float64` would remove the risk.
7. **Observability.** Structured logging exists. Prometheus metrics and traces
   do not.

---

## Data attribution

`intake_api/food.csv` comes from US Department of Agriculture food composition
data. Public domain.

`ml_api/nhanes.csv` comes from the US Centers for Disease Control and
Prevention, National Center for Health Statistics (NHANES 2013-2014). Public
domain, de-identified public-use survey data.

Both are works of the US Government and are not subject to copyright within the
United States.

---

## License

Released under the [MIT License](LICENSE).
