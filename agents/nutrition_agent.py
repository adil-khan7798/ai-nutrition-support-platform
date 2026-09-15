"""
nutrition_agent.py — Agent 1: Nutritional Analyzer (LLM agent + its tools).

Role (from the project summary):
    Retrieves a user's full intake history and identifies nutritional patterns
    and gaps.

This agent is an `LLMAgent` driving two tools:
    * compute_nutritional_gaps — the deterministic analysis (averages each
      tracked nutrient across all of the user's intake records and compares to
      FDA reference daily values). Authoritative numbers.
    * get_intake_records — the raw per-day records, so the model can inspect
      variability / data quality if it wants to.

The LLM decides which tools to call and writes the nutritional interpretation;
the arithmetic and the reference comparison stay deterministic in
`NutritionalAnalyzer` so the numbers it reasons over are always correct.

API used (inside the tools): GET /api/users/{user_id}/intake (via IntakeAPIClient).
"""

import logging
from typing import Dict, List, Optional

from clients import IntakeAPIClient
from context import WorkflowContext
from llm import NO_ARGS, LLMAgent, Tool

logger = logging.getLogger("agents.nutrition")

# ===========================================================================
# REFERENCE DAILY INTAKE
# ---------------------------------------------------------------------------
# FDA Daily Values (2,000 kcal reference diet), restricted to the 8 nutrients
# the platform actually tracks. `direction` says how to read the target:
#   "min"  — a floor; being consistently below it is a gap   (protein, fibre)
#   "max"  — a ceiling; being consistently above it is excess (sodium, fats…)
#   "range"— a rough target; flagged only when far from it    (carbohydrates)
# ===========================================================================
REFERENCE_DAILY_INTAKE: Dict[str, Dict] = {
    "protein_g":       {"target": 50.0,   "direction": "min",   "label": "protein"},
    "carbs_g":         {"target": 275.0,  "direction": "range", "label": "carbohydrates"},
    "sugar_g":         {"target": 50.0,   "direction": "max",   "label": "sugar"},
    "fiber_g":         {"target": 28.0,   "direction": "min",   "label": "fibre"},
    "total_fat_g":     {"target": 78.0,   "direction": "max",   "label": "total fat"},
    "saturated_fat_g": {"target": 20.0,   "direction": "max",   "label": "saturated fat"},
    "cholesterol_mg":  {"target": 300.0,  "direction": "max",   "label": "cholesterol"},
    "sodium_mg":       {"target": 2300.0, "direction": "max",   "label": "sodium"},
}

# How far past a "range" target counts as notably high / low.
RANGE_TOLERANCE = 0.5  # ±50 %


# ===========================================================================
# DAY WINDOWING
# ---------------------------------------------------------------------------
# The frontend lets a user analyse "the past X days" of intake. Records are
# grouped by their logged day (intake_date, falling back to the submission
# timestamp), and the most recent X distinct days are kept — mirroring how the
# Intake API itself windows its rolling weekly average, so a day still counts
# once even if it has several submissions.
# ===========================================================================
def _record_date(record: Dict) -> str:
    """The calendar day a record belongs to (intake_date, else timestamp's date)."""
    intake_date = record.get("intake_date")
    if intake_date:
        return str(intake_date)[:10]
    return str(record.get("timestamp", ""))[:10]


def _record_sort_key(record: Dict):
    return _record_date(record), str(record.get("timestamp", ""))


def window_records(records: List[Dict], days: Optional[int] = None) -> List[Dict]:
    """Keep only the records from the most recent `days` distinct logged days.

    `days` of None or <= 0 returns every record unchanged.
    """
    if not records or not days or days <= 0:
        return list(records)
    ordered = sorted(records, key=_record_sort_key)
    recent_days: List[str] = []
    for record in reversed(ordered):
        day = _record_date(record)
        if day not in recent_days:
            recent_days.append(day)
            if len(recent_days) >= days:
                break
    keep = set(recent_days)
    return [r for r in ordered if _record_date(r) in keep]


# ===========================================================================
# DETERMINISTIC ANALYSIS (tool backend)
# ===========================================================================
class NutritionalAnalyzer:
    """Turns raw intake history into a structured nutritional-gap report."""

    def __init__(self, intake_client: IntakeAPIClient) -> None:
        self.intake = intake_client

    def analyze(self, user_id: str, days: Optional[int] = None) -> Dict:
        """Fetch the user's intake history and return a structured gap report.

        `days` restricts the analysis to the most recent X logged days; None
        (the default) analyses the user's full history.
        """
        records = window_records(self.intake.get_intake_history(user_id), days)
        if not records:
            logger.warning("nutrition_no_records user=%s", user_id)
            return {
                "user_id": user_id,
                "records_analyzed": 0,
                "window_days": days,
                "averages": {},
                "below_target": [],
                "above_target": [],
                "within_target": [],
                "contributors": {},
                "note": "No intake records found for this user.",
            }

        averages = self._average_totals(records)
        below, above, within = self._compare_to_reference(averages)
        report = {
            "user_id": user_id,
            "records_analyzed": len(records),
            "window_days": days,
            "averages": averages,
            "below_target": below,
            "above_target": above,
            "within_target": within,
            "contributors": self._top_contributors(records),
        }
        logger.info(
            "nutrition_report user=%s records=%d window_days=%s below=%d above=%d",
            user_id, len(records), days, len(below), len(above),
        )
        return report

    @staticmethod
    def _average_totals(records: List[Dict]) -> Dict[str, float]:
        sums = {field: 0.0 for field in REFERENCE_DAILY_INTAKE}
        counts = {field: 0 for field in REFERENCE_DAILY_INTAKE}
        for record in records:
            totals = record.get("nutrient_totals", {}) or {}
            for field in REFERENCE_DAILY_INTAKE:
                if field in totals and totals[field] is not None:
                    sums[field] += float(totals[field])
                    counts[field] += 1
        return {
            field: round(sums[field] / counts[field], 2)
            for field in REFERENCE_DAILY_INTAKE
            if counts[field] > 0
        }

    @staticmethod
    def _compare_to_reference(averages: Dict[str, float]):
        below: List[Dict] = []
        above: List[Dict] = []
        within: List[Dict] = []

        for field, avg in averages.items():
            spec = REFERENCE_DAILY_INTAKE[field]
            target = spec["target"]
            pct = round(avg / target, 2) if target else None
            entry = {
                "field": field,
                "label": spec["label"],
                "average": avg,
                "target": target,
                "pct_of_target": pct,
                "delta": round(avg - target, 2),
            }
            direction = spec["direction"]
            if direction == "min":
                (below if avg < target else within).append(entry)
            elif direction == "max":
                (above if avg > target else within).append(entry)
            else:  # "range"
                if avg < target * (1 - RANGE_TOLERANCE):
                    below.append(entry)
                elif avg > target * (1 + RANGE_TOLERANCE):
                    above.append(entry)
                else:
                    within.append(entry)

        below.sort(key=lambda e: e["pct_of_target"] or 0)
        above.sort(key=lambda e: e["pct_of_target"] or 0, reverse=True)
        return below, above, within

    @staticmethod
    def _top_contributors(
        records: List[Dict], top_n: int = 3
    ) -> Dict[str, List[Dict]]:
        """Which logged foods contribute most to each nutrient over the window.

        Reads each record's `resolved_items` (the per-food scaled nutrients the
        Intake API stored) and sums every food's contribution to each nutrient.
        Returns, per nutrient field, the top `top_n` foods with the total they
        contributed and their share of that nutrient's total intake — so the
        risk agent can name the specific foods driving an over-target nutrient.
        """
        totals: Dict[str, float] = {f: 0.0 for f in REFERENCE_DAILY_INTAKE}
        per_food: Dict[str, Dict[str, float]] = {f: {} for f in REFERENCE_DAILY_INTAKE}
        for record in records:
            for item in record.get("resolved_items", []) or []:
                name = (
                    item.get("matched_description")
                    or item.get("submitted_name")
                    or "unknown food"
                )
                scaled = item.get("scaled_nutrients", {}) or {}
                for field in REFERENCE_DAILY_INTAKE:
                    amount = float(scaled.get(field, 0.0) or 0.0)
                    if amount <= 0:
                        continue
                    per_food[field][name] = per_food[field].get(name, 0.0) + amount
                    totals[field] += amount

        contributors: Dict[str, List[Dict]] = {}
        for field, foods in per_food.items():
            ranked = sorted(foods.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
            total = totals[field]
            contributors[field] = [
                {
                    "food": name,
                    "amount": round(amount, 1),
                    "share": round(amount / total, 2) if total else None,
                }
                for name, amount in ranked
            ]
        return contributors


# ===========================================================================
# LLM AGENT
# ===========================================================================
NUTRITION_SYSTEM_PROMPT = """\
You are the Nutritional Analyzer — Agent 1 of a two-agent dietary-health workflow.
Your job is to retrieve a user's intake history and identify their nutritional
patterns and gaps. You do NOT give medical advice or risk assessments; that is
Agent 2's role.

Tools:
- compute_nutritional_gaps: averages the user's nutrient intake across every
  logged day and compares it to FDA reference daily values. This returns the
  authoritative numbers — always call it, and base every figure you quote on it.
- get_intake_records: the raw per-day intake records. Call it only if you want
  to comment on how many days were logged, day-to-day variability, or foods that
  could not be matched.

Then write a concise, factual nutritional analysis (a short paragraph plus a few
bullet points). State which nutrients are consistently ABOVE reference (e.g.
sodium, saturated fat, sugar) and which are BELOW (e.g. fibre, protein), each
with its average vs. its target. Note any data caveats (few records, unmatched
foods). Never invent numbers that did not come from a tool. If there are no
intake records, say so plainly and stop.
"""


def build_nutrition_tools(
    user_id: str,
    intake: IntakeAPIClient,
    ctx: WorkflowContext,
    days: Optional[int] = None,
) -> List[Tool]:
    """Tools for Agent 1, bound to one user_id (so the model passes no arguments).

    `days` optionally restricts every tool to the user's most recent X logged days.
    """
    analyzer = NutritionalAnalyzer(intake)

    def get_intake_records() -> Dict:
        records = window_records(intake.get_intake_history(user_id), days)
        return {"user_id": user_id, "count": len(records), "records": records}

    def compute_nutritional_gaps() -> Dict:
        report = analyzer.analyze(user_id, days)
        ctx.nutrition_report = report  # captured for the orchestrator + Agent 2
        return report

    return [
        Tool(
            "compute_nutritional_gaps",
            "Average the user's nutrient intake across all logged days and compare "
            "to FDA reference daily values. Returns averages and which nutrients are "
            "below/above/within target, with exact numbers.",
            NO_ARGS,
            compute_nutritional_gaps,
        ),
        Tool(
            "get_intake_records",
            "Fetch the user's raw daily intake records (aggregated nutrient totals "
            "per logged day).",
            NO_ARGS,
            get_intake_records,
        ),
    ]


def build_nutrition_agent(
    user_id: str,
    intake: IntakeAPIClient,
    ctx: WorkflowContext,
    model: Optional[str] = None,
    client: object = None,
    days: Optional[int] = None,
) -> LLMAgent:
    return LLMAgent(
        name="nutritional-analyzer",
        system_prompt=NUTRITION_SYSTEM_PROMPT,
        tools=build_nutrition_tools(user_id, intake, ctx, days),
        model=model,
        client=client,
    )
