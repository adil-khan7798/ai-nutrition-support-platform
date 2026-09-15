"""
workflow.py — the two-agent (LLM) orchestrator + CLI.

Runs the agentic pipeline from the project summary, with both agents powered by
an OpenAI model using tool-calling:

    User intake history
            │
    [Agent 1 — Nutritional Analyzer]   (LLM)
        tools: compute_nutritional_gaps, get_intake_records
            │  nutritional analysis (prose) + structured gap report
            ▼   ── handoff ──
    [Agent 2 — Health Risk Reporter]   (LLM)
        tool: predict_health_risk  → Databricks MLflow /invocations
            │  plain-language health summary + lifestyle suggestions
            ▼
    Final output (not a medical diagnosis)

The Intake API supplies the data; the health-risk prediction comes from the
`health_food_risk_detector` model hosted on Databricks. The LLM agents decide
when to call their tools and turn the results into language; the tools do the
deterministic work (averaging, the 106-column feature record, the model call).

USAGE
    # config via env (.env — see .env.example), then:
    python workflow.py --user-id <USER_ID>

    Required env: OPENAI_API_KEY, DATABRICKS_TOKEN.
    Optional:     OPENAI_MODEL (default gpt-4o), INTAKE_API_URL, DATABRICKS_ENDPOINT_URL.

    --json prints the full structured result (reports + prediction + summary).
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, Optional

from dotenv import load_dotenv

from clients import DatabricksRiskClient, IntakeAPIClient
from context import WorkflowContext
from nutrition_agent import NutritionalAnalyzer, build_nutrition_agent
from risk_agent import DISCLAIMER, build_risk_agent

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("agents.workflow")


def run_workflow(
    user_id: str,
    intake_client: IntakeAPIClient,
    risk_client: DatabricksRiskClient,
    model: Optional[str] = None,
    llm_client: object = None,
    days: Optional[int] = None,
    stored_risk_context: Optional[str] = None,
) -> Dict:
    """Run Agent 1 (LLM) → Agent 2 (LLM) for one user and return the result.

    `days` restricts both agents to the user's most recent X logged days; None
    analyses the full intake history.
    """
    if llm_client is None:
        from openai import OpenAI  # lazy: tests inject a fake client

        llm_client = OpenAI()

    ctx = WorkflowContext()
    window_note = (
        f" Consider only their most recent {days} logged days of intake."
        if days else ""
    )

    # --- Agent 1: Nutritional Analyzer ---
    agent1 = build_nutrition_agent(
        user_id, intake_client, ctx, model, llm_client, days=days
    )
    nutrition_analysis = agent1.run(
        f"Analyse the dietary intake history for user '{user_id}' and report their "
        f"nutritional patterns and gaps.{window_note}"
    )

    # Guarantee downstream has the exact numbers even if the model skipped the tool.
    if ctx.nutrition_report is None:
        ctx.nutrition_report = NutritionalAnalyzer(intake_client).analyze(user_id, days)

    # --- Agent 2: Health Risk Reporter (handoff of Agent 1's findings) ---
    agent2 = build_risk_agent(
        user_id, intake_client, risk_client, ctx, model, llm_client, days=days
    )
    stored_note = (
        f"\n\nIMPORTANT CONTEXT: {stored_risk_context} "
        "Treat this as the primary risk signal when writing your summary — "
        "if it indicates elevated risk, your summary must reflect that clearly."
    ) if stored_risk_context else ""

    summary = agent2.run(
        f"User '{user_id}'. The Nutritional Analyzer (Agent 1) reported:\n\n"
        f"{nutrition_analysis}\n\n"
        f"Call predict_health_risk, then COMBINE the model's risk result with "
        f"these nutritional findings into the final plain-language health "
        f"summary with lifestyle suggestions for this user.{stored_note}"
    )

    # Safety net: the disclaimer is mandatory even if the model forgets it.
    if DISCLAIMER not in summary:
        summary = (summary.rstrip() + "\n\n" + DISCLAIMER).strip()

    return {
        "user_id": user_id,
        "window_days": days,
        "nutrition_report": ctx.nutrition_report,
        "nutrition_analysis": nutrition_analysis,
        "prediction": ctx.prediction,
        "summary": summary,
    }


def _build_clients(args) -> tuple[IntakeAPIClient, DatabricksRiskClient]:
    intake = IntakeAPIClient(base_url=args.intake_url, token=args.intake_token)
    if not intake.token:
        # Default to the `service` account: under the Intake API's resource-based
        # authorization a `user` token may only read its own record, but this
        # workflow reads arbitrary --user-id values, which needs the service role.
        username = args.username or os.environ.get("INTAKE_USERNAME", "service")
        password = args.password or os.environ.get("INTAKE_PASSWORD", "service-password")
        intake.login(username, password)

    risk = DatabricksRiskClient(
        endpoint_url=args.databricks_url, token=args.databricks_token
    )
    return intake, risk


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Two-agent (LLM) diet → health-risk workflow, Databricks-backed."
    )
    parser.add_argument("--user-id", required=True, help="User ID to analyse")
    parser.add_argument(
        "--intake-url",
        default=os.environ.get("INTAKE_API_URL"),
        help="Intake API base URL (default env INTAKE_API_URL or http://localhost:8001)",
    )
    parser.add_argument(
        "--intake-token", default=None, help="Pre-issued Intake API JWT (skips login)"
    )
    parser.add_argument("--username", default=None, help="Intake API login username")
    parser.add_argument("--password", default=None, help="Intake API login password")
    parser.add_argument(
        "--databricks-url",
        default=os.environ.get("DATABRICKS_ENDPOINT_URL"),
        help="Databricks MLflow /invocations URL (default: health-food-risk endpoint)",
    )
    parser.add_argument(
        "--databricks-token",
        default=os.environ.get("DATABRICKS_TOKEN"),
        help="Databricks bearer token (default env DATABRICKS_TOKEN)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL"),
        help="OpenAI model for the agents (default env OPENAI_MODEL or gpt-4o)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Analyse only the user's most recent N logged days (default: all history)",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the full structured result as JSON"
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is not set (add it to agents/.env).")
        return 1

    try:
        intake_client, risk_client = _build_clients(args)
        result = run_workflow(
            args.user_id, intake_client, risk_client, model=args.model, days=args.days
        )
    except Exception as e:  # noqa: BLE001 — CLI boundary: surface a clean message
        logger.error("workflow_failed: %s", e)
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n=== Agent 1 — Nutritional Analysis ===\n")
        print(result["nutrition_analysis"])
        print("\n=== Agent 2 — Health Summary ===\n")
        print(result["summary"])
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
