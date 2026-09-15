# Frontend - Nutrition Support Platform (Streamlit)

A Streamlit client that connects the platform's pieces: the Intake API (auth,
profiles, food search, intake), the per-day ML predictions, and the two-agent
LLM health-analysis workflow.

## What it does

| Tab | Backed by |
|-----|-----------|
| **Profile** | `POST /v1/login`, `POST /users` |
| **Log intake** | `GET /foods/search`, `POST /users/{id}/intake` (+ a "Seed last 7 days" button) |
| **Dashboard** | `GET /users/{id}/intake` → avg-vs-target, nutrient trends, risk-over-time charts |
| **Health analysis** | the agentic workflow (`agents/workflow.py`) over the most recent **N days** |

The "Health analysis" tab lets you choose how many recent days to analyse
(default 7). The window is threaded through the agents so both the nutrient-gap
analysis **and** the risk model's feature record use only that window. The
"Seed last 7 days" button auto-populates a week of real foods so you can run the
analysis immediately.

## Prerequisites

1. **Intake API** running and reachable (default `http://localhost:8001`):
   ```
   cd intake_api && python intake_api.py
   ```
2. *(optional)* **ML API** on `:8002` if you want per-day risk predictions stored
   with each intake record (`cd ml_api && python ml_api.py`).
3. *(optional, for the Health analysis tab)* `OPENAI_API_KEY` and
   `DATABRICKS_TOKEN` in `agents/.env`.

## Run

```bash
pip install -r frontend/requirements.txt
streamlit run frontend/app.py     # opens http://localhost:8501
```

Streamlit's default port (8501) is already in the Intake API's CORS allow-list.

## Configuration

| Env var | Default | Used by |
|---------|---------|---------|
| `INTAKE_API_URL` | `http://localhost:8001` | the Intake API client (also editable in the sidebar) |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | — / `gpt-4o` | the agentic analysis (from `agents/.env`) |
| `DATABRICKS_TOKEN`, `DATABRICKS_ENDPOINT_URL` | — / hosted endpoint | the risk model call |

## Demo flow

1. Log in from the sidebar (`user` / `user-password`, or set `DEMO_PASSWORD`).
2. **Profile** → Register profile.
3. **Log intake** → *Seed last 7 days of intake*.
4. **Dashboard** → inspect trends and averages vs target.
5. **Health analysis** → pick a window (e.g. 7 days) → *Run health analysis*.
