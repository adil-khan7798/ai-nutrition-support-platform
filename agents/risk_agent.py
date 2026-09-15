"""
risk_agent.py — Agent 2: Health Risk Reporter (LLM agent + its tools).

Role (from the project summary):
    Combines the Nutritional Analyzer's output with the ML model's risk
    prediction to produce a plain-language health summary and general lifestyle
    suggestions.

This agent is an `LLMAgent` with a single tool, `predict_health_risk`, which
runs the Databricks-hosted model and hands back the label. The LLM receives
Agent 1's nutritional findings in its task prompt, calls the tool, and writes
the final summary — connecting the user's specific nutritional patterns to the
predicted risk and offering general lifestyle suggestions, always closing with
a not-a-diagnosis disclaimer (also enforced in code by the orchestrator).

ML model:
    `workspace.dataattproject.health_food_risk_detector` — a Spark ML pipeline
    (RandomForest, 100 trees, max depth 8) hosted on Databricks and served via
    MLflow as a REST endpoint. It returns a single binary label,
    `high_health_risk_label` (1 = obese and/or high blood pressure → elevated
    risk, 0 = lower risk).
"""

import logging
from typing import Dict, List, Optional

from clients import DatabricksRiskClient, DatabricksRiskError, IntakeAPIClient
from context import WorkflowContext
from feature_schema import FEATURE_ORDER
from llm import NO_ARGS, LLMAgent, Tool
from nutrition_agent import NutritionalAnalyzer

logger = logging.getLogger("agents.risk")

DISCLAIMER = (
    "This is not a medical diagnosis — please consult a qualified healthcare "
    "professional for personalised medical advice."
)


# ===========================================================================
# FEATURE RECORD — the single place that assembles the model input
# ---------------------------------------------------------------------------
# The model's signature is the 106 columns in feature_schema.FEATURE_ORDER.
# The platform can source only a handful of them (demographics + the 8 nutrient
# totals the Intake API aggregates, all named identically to the model). Every
# other column — labs, income-to-poverty, waist, calories, caffeine — is sent
# as null and filled by the pipeline's median Imputer, exactly as the notebook
# intends ("the model can still run if some lab values are missing").
#
# NOTE: bmi and the blood-pressure columns are deliberately NOT here — they
# were excluded from training as label-leakage, so they are not model inputs.
# ===========================================================================
_PROFILE_FIELDS = ("gender", "age", "race_ethnicity", "education_level",
                   "weight_kg", "height_cm")
_INTEGER_FIELDS = {"gender", "race_ethnicity", "education_level", "medication_count"}


def build_feature_record(
    profile: Dict, averages: Dict[str, float]
) -> Dict[str, Optional[float]]:
    """Assemble the 106-column feature row sent to the Databricks model.

    Returns a dict keyed by FEATURE_ORDER. Known columns are filled from the
    user's profile and *averaged* daily nutrient intake (Agent 1's output);
    every unknown column is `None` so the pipeline's median Imputer fills it.
    """
    known: Dict[str, Optional[float]] = {}

    for field in _PROFILE_FIELDS:
        if profile.get(field) is not None:
            known[field] = profile[field]

    for field, value in averages.items():
        if field in FEATURE_ORDER:
            known[field] = value

    # New users have no medication history; the ETL fills this with 0, not null.
    known["medication_count"] = 0

    record: Dict[str, Optional[float]] = {}
    for col in FEATURE_ORDER:
        value = known.get(col)
        if value is None:
            record[col] = None                       # imputed at inference time
        elif col in _INTEGER_FIELDS:
            record[col] = int(value)
        else:
            record[col] = float(value)
    return record


# ===========================================================================
# LLM AGENT
# ===========================================================================
RISK_SYSTEM_PROMPT = f"""\
You are the Health Risk Reporter — Agent 2 of a two-agent dietary-health
workflow. You are given the Nutritional Analyzer's (Agent 1) findings for a
user. Your job is to COMBINE those findings with the ML model's risk prediction
into a single plain-language health summary with general lifestyle suggestions.

Tool:
- predict_health_risk: scores the user's demographic profile and Agent 1's
  averaged nutrient intake against the Databricks-hosted
  `health_food_risk_detector` model. Call it exactly once. It returns:
    * high_health_risk_label — 1 = elevated risk (associated with obesity
      and/or high blood pressure), 0 = lower risk;
    * nutrients_above_target / nutrients_below_target — Agent 1's flagged
      nutrients, each with its average, reference target, and top_foods: the
      specific foods from the user's OWN logs that contributed most to that
      nutrient (each with its share of the total).

Then write a warm, encouraging, plain-language summary (a short paragraph) that
COMBINES both inputs and is grounded in the user's actual food logs:
1. state the model's result in lay terms (elevated vs. lower risk);
2. name the user's SPECIFIC flagged nutrients (e.g. sodium consistently above,
   fibre below) and tie each to a general, evidence-informed health area —
   high sodium ↔ blood pressure and cardiovascular health; low fibre ↔
   digestive health and cholesterol; high saturated fat/cholesterol ↔
   cardiovascular health; low protein ↔ muscle maintenance and satiety;
3. for each OVER-target nutrient, name the specific foods from its top_foods
   (foods the user actually logged) that are driving it, briefly explain WHY
   those foods are high in that nutrient, and suggest cutting back on or
   swapping them (e.g. "the bacon and cheese in your logs are high in sodium
   and saturated fat — try leaner or lower-salt swaps");
4. for each UNDER-target nutrient, suggest specific, easy-to-add foods rich in
   it (low fibre → whole grains, legumes, vegetables; low protein → lean meat,
   eggs, beans).

Expected shape of the combined output:
"Based on your recent intake logs, your sodium intake is consistently above the
general recommended level — driven mainly by the bacon and cheese you've been
eating — while your fibre is below it. The health-risk model flags your profile
as elevated risk, and these patterns are associated with cardiovascular and
digestive health. Consider cutting back on the bacon and cheese and adding more
whole grains, legumes, and vegetables to raise your fibre. {DISCLAIMER}"

Safety rules (must follow):
- This is general wellness information, NOT a medical diagnosis. Never state or
  imply the user has any disease; use "associated with" / "may be linked to".
- Do not invent nutrient numbers or risks beyond what Agent 1 and the tool gave.
- Only name foods that appear in a nutrient's top_foods list — never invent
  foods the user did not log.
- If predict_health_risk returns an error, say the risk estimate is currently
  unavailable and base the summary on the nutritional findings alone.
- End your summary with EXACTLY this sentence on its own:
  "{DISCLAIMER}"
"""


def build_risk_tools(
    user_id: str,
    intake: IntakeAPIClient,
    databricks: DatabricksRiskClient,
    ctx: WorkflowContext,
    days: Optional[int] = None,
) -> List[Tool]:
    """Tools for Agent 2, bound to one user_id.

    `days` optionally restricts the nutrient averages (and therefore the model's
    feature record) to the user's most recent X logged days.
    """
    analyzer = NutritionalAnalyzer(intake)

    def predict_health_risk() -> Dict:
        if ctx.profile is None:
            ctx.profile = intake.get_user(user_id)["profile"]
        if ctx.nutrition_report is None:
            ctx.nutrition_report = analyzer.analyze(user_id, days)

        averages = ctx.nutrition_report.get("averages", {})
        features = build_feature_record(ctx.profile, averages)
        ctx.features = features

        try:
            prediction = databricks.predict(features)
        except DatabricksRiskError as e:
            logger.warning("risk_predict_failed user=%s: %s", user_id, e)
            return {"error": f"risk model unavailable: {e}"}

        ctx.prediction = prediction
        n_known = sum(1 for v in features.values() if v is not None)
        logger.info(
            "risk_prediction user=%s label=%s prob=%s",
            user_id, prediction["label"], prediction["probability"],
        )

        report = ctx.nutrition_report or {}
        contributors = report.get("contributors", {})

        # Hand Agent 1's flagged patterns back alongside the model's risk so the
        # agent cross-references the two (spec step 3) from structured data,
        # rather than re-extracting nutrients from Agent 1's prose. Each flagged
        # nutrient carries the specific foods from the user's logs that drove it,
        # so the summary can name them and recommend cutting back / swapping.
        def _flags(entries: List[Dict]) -> List[Dict]:
            flags = []
            for e in entries:
                flag = {
                    "nutrient": e["label"],
                    "average": e["average"],
                    "target": e["target"],
                }
                foods = contributors.get(e["field"], [])
                if foods:
                    flag["top_foods"] = [
                        {"food": f["food"], "share": f["share"]} for f in foods
                    ]
                flags.append(flag)
            return flags

        return {
            "high_health_risk_label": prediction["label"],
            "risk": "elevated" if prediction["label"] == 1 else "lower",
            "probability": prediction["probability"],
            "label_meaning": "1 = elevated risk (obesity and/or high blood "
                             "pressure); 0 = lower risk",
            "nutrients_above_target": _flags(report.get("above_target", [])),
            "nutrients_below_target": _flags(report.get("below_target", [])),
            "features_provided": n_known,
            "features_imputed": len(features) - n_known,
        }

    return [
        Tool(
            "predict_health_risk",
            "Run the Databricks health-risk model on the user's profile and "
            "averaged intake. Returns high_health_risk_label (1 = elevated, "
            "0 = lower) together with Agent 1's flagged nutrients "
            "(nutrients_above_target / nutrients_below_target), each annotated "
            "with the specific foods from the user's logs that drove it.",
            NO_ARGS,
            predict_health_risk,
        ),
    ]


def build_risk_agent(
    user_id: str,
    intake: IntakeAPIClient,
    databricks: DatabricksRiskClient,
    ctx: WorkflowContext,
    model: Optional[str] = None,
    client: object = None,
    days: Optional[int] = None,
) -> LLMAgent:
    return LLMAgent(
        name="health-risk-reporter",
        system_prompt=RISK_SYSTEM_PROMPT,
        tools=build_risk_tools(user_id, intake, databricks, ctx, days),
        model=model,
        client=client,
    )
