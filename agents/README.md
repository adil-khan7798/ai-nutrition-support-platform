# Agentic Workflow — Diet → Health-Risk Summary

A two-agent pipeline that turns a user's logged dietary intake into a
plain-language health summary. Both agents are **LLM agents** (OpenAI, via
tool-calling): the model decides when to call its tools and writes the prose,
while the tools do the precise/deterministic work — fetching data, the nutrient
math, and the Databricks ML call. It is wired to the ML model hosted on **Databricks** and served through
**MLflow** as a REST endpoint.

```
User intake history
        │
[Agent 1 — Nutritional Analyzer]  (LLM)
   tools: compute_nutritional_gaps · get_intake_records   → GET /api/users/{id}/intake
        │  nutritional analysis (prose) + structured gap report
        ▼   ── handoff ──
[Agent 2 — Health Risk Reporter]  (LLM)
   tool:  predict_health_risk   → Databricks MLflow /invocations
        │  plain-language health summary + lifestyle suggestions
        ▼
Final output (not a medical diagnosis — for informational purposes only)
```

## The two agents

### Agent 1 — Nutritional Analyzer (`nutrition_agent.py`)
An LLM agent with two tools:
- `compute_nutritional_gaps` — averages each tracked nutrient across **all** of
  the user's intake records and compares to **FDA reference daily values**,
  returning which nutrients are below/above/within target with exact numbers
  (deterministic; reads `GET /api/users/{user_id}/intake`).
- `get_intake_records` — the raw per-day records, for commenting on data volume
  or variability.

The model calls the tools and writes a factual nutritional analysis; every
number it quotes comes from a tool, never from the model itself.

### Agent 2 — Health Risk Reporter (`risk_agent.py`)
An LLM agent with one tool:
- `predict_health_risk` — assembles the model's **106-column feature record**
  and scores it against the **`workspace.dataattproject.health_food_risk_detector`**
  Spark ML pipeline (RandomForest) on Databricks. It returns the binary
  `high_health_risk_label` (`1` = obese and/or high blood pressure → elevated
  risk; `0` = lower risk) **together with Agent 1's flagged nutrients**
  (above/below target), so the agent cross-references risk and nutrition from
  structured data.

It receives Agent 1's findings, calls the tool, and writes the plain-language
summary — **combining** the model's risk result with the user's specific
flagged nutritional patterns and tying them to general health areas and
lifestyle suggestions — always closing with a *not-a-diagnosis* disclaimer
(also enforced in code by the orchestrator).

## Files

| File | Purpose |
|------|---------|
| `llm.py` | Generic OpenAI tool-calling agent runtime (`LLMAgent`, `Tool`). |
| `clients.py` | `IntakeAPIClient` (login + reads) and `DatabricksRiskClient` (MLflow REST). |
| `context.py` | `WorkflowContext` — per-run scratchpad where tools record structured results. |
| `nutrition_agent.py` | Agent 1 — the deterministic analyzer + its tools + system prompt. |
| `risk_agent.py` | Agent 2 — the 106-column feature record + its tool + system prompt. |
| `feature_schema.py` | The model's exact 106-column input signature, auto-generated from the ETL notebook. |
| `workflow.py` | Orchestrator that runs Agent 1 → Agent 2, plus the CLI. |
| `.env.example` | Configuration template (copy to `.env`). |

## Configuration

Copy `.env.example` to `.env` and fill it in (`.env` is gitignored):

| Variable | Meaning |
|----------|---------|
| `OPENAI_API_KEY` | **Required.** Powers both LLM agents. |
| `OPENAI_MODEL` | Tool-calling chat model for the agents (default `gpt-4o`). |
| `INTAKE_API_URL` | Intake API base. `http://localhost:8001` locally, or `https://<host>/api` through the ingress. |
| `INTAKE_USERNAME` / `INTAKE_PASSWORD` | Credentials used to obtain a JWT from the Intake API. Default to the `service` account — resource-based authorization lets a `user` token read only its own record, but the workflow reads arbitrary user IDs. |
| `DATABRICKS_ENDPOINT_URL` | The MLflow `/invocations` URL (defaults to the health-food-risk endpoint). |
| `DATABRICKS_TOKEN` | **Required.** Databricks PAT or service-principal token, sent as a Bearer credential. Never commit it. |

## Run

```bash
pip install -r requirements.txt

# Make sure OPENAI_API_KEY and DATABRICKS_TOKEN are set (in .env), then:
python workflow.py --user-id <USER_ID>

# Full structured result (both reports + prediction + summary) as JSON:
python workflow.py --user-id <USER_ID> --json
```

Every config value also has a CLI flag (`--model`, `--intake-url`,
`--databricks-token`, `--username`, …) which overrides the environment. Run
`python workflow.py -h` for the full list. If you already hold an Intake API JWT,
pass `--intake-token` to skip the login step.

## How the agents work (tool-calling loop)

`llm.py` runs the standard agentic loop for each agent: send the conversation to
the model advertising the agent's tools → if the model requests tool calls,
execute them and feed the JSON results back → repeat until the model returns a
plain-text answer (bounded by `max_steps`). The tools are bound to a single
`user_id` at construction, so the model never has to pass (or hallucinate) IDs.

The orchestrator (`workflow.py`) runs Agent 1, hands its written analysis to
Agent 2 in the task prompt, then runs Agent 2. Both agents share one
`WorkflowContext`, so the exact numbers the tools produced (the gap report, the
prediction, the feature record actually sent) are available for the `--json`
output — independent of the model's prose. The mandatory disclaimer is enforced
in code as a safety net even if the model omits it.

## How the Databricks call works

The model is the Spark ML pipeline registered by the ETL notebook
(`Data Ingestion.ipynb`). Its logged signature is the **106 numeric
columns** in `feature_schema.FEATURE_ORDER` — demographics, nutrition,
`medication_count`, body measures, and ~87 `lab_*` columns. `bmi` and the
blood-pressure columns are **not** inputs: they were dropped from training as
label-leakage (the label is derived from them).

`DatabricksRiskClient` POSTs the MLflow `dataframe_records` payload — the same
shape the `ml-api` service sends to this endpoint (`ml_api.py`) — covering all
106 columns:

```json
{
  "dataframe_records": [
    {"gender": 1, "age": 54.0, "...": "...", "sodium_mg": 3400.0, "lab_URDFLOW1": null}
  ]
}
```

with `Authorization: Bearer <DATABRICKS_TOKEN>`, and reads back
`{"predictions": [...]}` — the Spark `prediction` column, a `0.0`/`1.0` label
with no probability. The response parser normalises whatever shape MLflow emits
(bare scalar, one-element list, or a dict) into `{label, probability}`. The
client allows a 120-second timeout to absorb the serving endpoint's cold-start
latency, matching the ml-api service.

> **Feature coverage.** The platform can only source a handful of those 106
> columns — the demographic profile plus the 8 nutrient totals the Intake API
> aggregates (named identically to the model). Every other column (labs,
> `income_to_poverty_ratio`, `waist_cm`, `calories`, `caffeine_mg`) is sent as
> `null` and filled by the pipeline's **median Imputer** at inference time —
> exactly the behaviour the notebook describes. `risk_agent.build_feature_record()`
> is the single place that does this mapping; widen it if the platform starts
> collecting more of these fields.
>
> **Integer-null caveat.** When the model was logged, MLflow warned that the
> inferred signature contains integer columns, which cannot represent missing
> values. The columns we fill (`gender`, `race_ethnicity`, `education_level`,
> `medication_count`) are always sent as real integers, so they are fine; but
> if any *unfilled* column was typed as integer, schema enforcement may reject
> its `null`. The fix is on the model side — re-log the signature with those
> columns as doubles (`float64`), as the MLflow warning itself recommends.

## Notes / limitations

- The **reasoning and prose are LLM-generated**; the **facts are not**. All
  numbers come from deterministic tools (the nutrient math and the ML call), so
  the model interprets and explains but never fabricates figures. The disclaimer
  is enforced in code regardless of model output.
- Calls cost OpenAI tokens. The model is configurable via `OPENAI_MODEL` /
  `--model`; a typical run is a handful of short tool-calling turns per agent.
- Reference values are FDA Daily Values (2,000 kcal diet) for the 8 nutrients
  the platform tracks — informational, not personalised medical targets.
- Intake data lives in the Intake API's in-memory store, so a user must have
  registered and logged intake in the **same** running instance.
