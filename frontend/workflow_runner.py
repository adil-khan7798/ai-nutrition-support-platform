"""workflow_runner.py — bridge from the Streamlit frontend to the agents package.

It puts the sibling `agents/` directory on sys.path, loads its `.env` (so the
OpenAI + Databricks credentials the workflow needs are available), and exposes:

  * run_analysis(...)            — run the two-agent LLM workflow over the last N days
  * gap_report_from_records(...) — the deterministic avg-vs-target report, for charts
  * NUTRIENT_LABELS              — display labels/units for the 8 tracked nutrients

The deterministic gap report reuses the exact same averaging + reference logic
the agents use (`NutritionalAnalyzer`), so the dashboard charts and the agent's
numbers can never drift apart.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# --- make the agents package importable + load its credentials --------------
_FRONTEND_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FRONTEND_DIR.parent           # the base/project directory
_AGENTS_DIR = _REPO_ROOT / "agents"
if str(_AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENTS_DIR))

from dotenv import load_dotenv  # noqa: E402

# Load credentials from agents/.env first, then fall back to the base directory's
# .env (and an optional frontend/.env) for anything not already set — load_dotenv
# does not override variables that an earlier file already provided.
for _env_path in (_AGENTS_DIR / ".env", _REPO_ROOT / ".env", _FRONTEND_DIR / ".env"):
    load_dotenv(_env_path)

from nutrition_agent import NutritionalAnalyzer, window_records  # noqa: E402

NUTRIENT_LABELS: Dict[str, str] = {
    "protein_g": "Protein (g)",
    "carbs_g": "Carbohydrates (g)",
    "sugar_g": "Sugar (g)",
    "fiber_g": "Fibre (g)",
    "total_fat_g": "Total fat (g)",
    "saturated_fat_g": "Saturated fat (g)",
    "cholesterol_mg": "Cholesterol (mg)",
    "sodium_mg": "Sodium (mg)",
}


class AnalysisUnavailable(RuntimeError):
    """Raised when the agentic analysis cannot run (e.g. missing credentials)."""


def missing_credentials() -> List[str]:
    """Return the names of any env vars the agentic workflow needs but lacks."""
    missing = []
    if not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.environ.get("DATABRICKS_TOKEN"):
        missing.append("DATABRICKS_TOKEN")
    return missing


def gap_report_from_records(
    records: List[Dict], days: Optional[int] = None
) -> Dict:
    """Deterministic avg-vs-target report from already-fetched intake records.

    Mirrors `NutritionalAnalyzer.analyze` but works on records the dashboard has
    already loaded (no extra API round-trip), windowed to the most recent X days.
    """
    windowed = window_records(records, days)
    averages = NutritionalAnalyzer._average_totals(windowed)
    below, above, within = NutritionalAnalyzer._compare_to_reference(averages)
    return {
        "records_analyzed": len(windowed),
        "window_days": days,
        "averages": averages,
        "below_target": below,
        "above_target": above,
        "within_target": within,
    }


def run_analysis(
    user_id: str,
    token: str,
    days: Optional[int] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    stored_risk_context: Optional[str] = None,
) -> Dict:
    """Run the two-agent workflow for one user over their most recent X days.

    Uses the caller's own JWT to read their records, and the Databricks endpoint
    (via DATABRICKS_TOKEN) for the risk model. Raises AnalysisUnavailable when a
    required credential is missing, so the UI can show a friendly message.
    """
    missing = missing_credentials()
    if missing:
        raise AnalysisUnavailable(
            "Missing required credentials: "
            + ", ".join(missing)
            + ". Add them to agents/.env to enable the health analysis."
        )

    # Imported lazily so the dashboard/charts work even without openai installed.
    from clients import DatabricksRiskClient, IntakeAPIClient
    from workflow import run_workflow

    intake = IntakeAPIClient(
        base_url=base_url or os.environ.get("INTAKE_API_URL"), token=token
    )
    risk = DatabricksRiskClient()  # reads DATABRICKS_TOKEN from the environment
    return run_workflow(
        user_id,
        intake,
        risk,
        model=model or os.environ.get("OPENAI_MODEL"),
        days=days,
        stored_risk_context=stored_risk_context,
    )
