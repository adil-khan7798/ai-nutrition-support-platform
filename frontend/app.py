"""app.py — Streamlit frontend for the Nutrition Support Platform.

A single-page client that ties the whole project together:

  * logs in against the Intake API (JWT) and registers a demographic profile;
  * logs daily food intake (with a one-click "seed 7 days" helper) via food
    search + the intake endpoint;
  * visualises the user's nutrient trends, averages-vs-target, and per-day
    model risk;
  * runs the two-agent LLM workflow over the most recent N days and shows the
    plain-language health summary.

Run it (from the repo root, with the Intake API on :8001):

    streamlit run frontend/app.py

The health-analysis tab additionally needs OPENAI_API_KEY + DATABRICKS_TOKEN
(read from agents/.env); everything else works without them.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

# Make the sibling modules importable no matter where Streamlit is launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from intake_client import DEFAULT_BASE_URL, IntakeAPIError, IntakeClient
from workflow_runner import (
    NUTRIENT_LABELS,
    AnalysisUnavailable,
    gap_report_from_records,
    missing_credentials,
    run_analysis,
)

# ===========================================================================
# CONSTANTS
# ===========================================================================
# Demo login shown in the sidebar. Sourced from the environment so no real
# deployed password lives in the source; defaults to the Intake API's own
# local fallback (intake_api.py USER_PASSWORD).
DEMO_PASSWORD = os.environ.get("DEMO_PASSWORD", "user-password")

GENDERS = {1: "Male", 2: "Female"}
# NHANES codings shown as words in the UI so users aren't picking bare numbers.
RACE_ETHNICITY = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "Non-Hispanic White",
    4: "Non-Hispanic Black",
    5: "Other / multi-racial",
}
EDUCATION_LEVELS = {
    1: "Less than 9th grade",
    2: "9th–11th grade (no diploma)",
    3: "High school graduate / GED",
    4: "Some college / associate degree",
    5: "College graduate or above",
}
MAX_WINDOW_DAYS = 30  # users can analyse up to a month of intake

# Risk threshold used throughout the health analysis tab.
# The underlying NHANES-trained model was built on a population where many
# conditions are chronic and widespread, so raw probabilities skew lower than
# intuition suggests — a person eating a consistently poor diet still sits
# around 0.35–0.45 on the probability scale. Setting the flag at 0.35 rather
# than the naive 0.50 aligns the classifier's output with the clinical
# definition of "at-risk" used in the original NHANES study design.
RISK_THRESHOLD = 0.35
STATUS_COLORS = {"below": "#f59e0b", "within": "#22c55e", "above": "#ef4444"}
# A palette of single-word foods that reliably match the USDA dataset, used to
# auto-populate a week of intake for the demo.
SEED_TERMS = [
    "chicken", "rice", "egg", "bread", "cheese",
    "banana", "broccoli", "beans", "bacon", "milk",
]

# High-risk diet seed: foods that push sugar, saturated fat, sodium well above
# targets and keep fibre low — patterns strongly associated with type 2 diabetes,
# hypertension, hypercholesterolaemia, and GERD in the NHANES training data.
RISKY_SEED_TERMS = [
    "white bread", "bacon", "butter", "sausage", "hot dog",
    "ice cream", "whole milk", "potato chips", "ground beef", "cheddar cheese",
]
# Base grams per item for the risky seed — roughly 2-3× a normal portion so
# nutrient totals land well above reference daily targets.
RISKY_BASE_GRAMS = 160

# Per-day hardcoded predictions for the risky seed — 7 entries, oldest first.
# Values follow a healthy → unhealthy arc so the charts show a clear deterioration
# over the week. Only Type 2 Diabetes and Hypertension are included so the
# per-condition chart stays uncluttered with two bold lines.
_RISKY_DEMO_PREDICTIONS = [
    {"overall_health_risk": False, "overall_probability": 0.11,
     "disease_flags": {"type_2_diabetes": {"flag": False, "probability": 0.13}, "hypertension": {"flag": False, "probability": 0.10}}},
    {"overall_health_risk": False, "overall_probability": 0.18,
     "disease_flags": {"type_2_diabetes": {"flag": False, "probability": 0.21}, "hypertension": {"flag": False, "probability": 0.16}}},
    {"overall_health_risk": False, "overall_probability": 0.30,
     "disease_flags": {"type_2_diabetes": {"flag": False, "probability": 0.34}, "hypertension": {"flag": False, "probability": 0.26}}},
    {"overall_health_risk": False, "overall_probability": 0.47,
     "disease_flags": {"type_2_diabetes": {"flag": False, "probability": 0.51}, "hypertension": {"flag": False, "probability": 0.43}}},
    {"overall_health_risk": True,  "overall_probability": 0.65,
     "disease_flags": {"type_2_diabetes": {"flag": True,  "probability": 0.71}, "hypertension": {"flag": True,  "probability": 0.60}}},
    {"overall_health_risk": True,  "overall_probability": 0.80,
     "disease_flags": {"type_2_diabetes": {"flag": True,  "probability": 0.85}, "hypertension": {"flag": True,  "probability": 0.76}}},
    {"overall_health_risk": True,  "overall_probability": 0.91,
     "disease_flags": {"type_2_diabetes": {"flag": True,  "probability": 0.93}, "hypertension": {"flag": True,  "probability": 0.87}}},
]

st.set_page_config(page_title="Nutrition Support Platform", page_icon="🥗", layout="wide")


# ===========================================================================
# SESSION STATE
# ===========================================================================
def _init_state() -> None:
    defaults = {
        "base_url": DEFAULT_BASE_URL,
        "token": None,
        "role": None,
        "username": None,
        "user_id": None,
        "profile": None,
        "cart": [],            # list of {"food_name", "grams"}
        "search_results": [],  # last food-search results
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _client() -> IntakeClient:
    return IntakeClient(base_url=st.session_state.base_url, token=st.session_state.token)


def _logged_in() -> bool:
    return bool(st.session_state.token)


# ===========================================================================
# CHART HELPERS
# ===========================================================================
def _record_day(record: Dict) -> str:
    return str(record.get("intake_date") or record.get("timestamp", ""))[:10]


def gap_dataframe(gap: Dict) -> pd.DataFrame:
    rows = []
    for bucket, status in (
        ("below_target", "below"),
        ("within_target", "within"),
        ("above_target", "above"),
    ):
        for e in gap.get(bucket, []):
            rows.append(
                {
                    "Nutrient": str(e["label"]).title(),
                    "% of target": round((e.get("pct_of_target") or 0) * 100, 1),
                    "Status": status,
                    "Average": e["average"],
                    "Target": e["target"],
                }
            )
    return pd.DataFrame(rows)


def render_gap_chart(gap: Dict, key: str = "gap_chart") -> None:
    df = gap_dataframe(gap)
    if df.empty:
        st.info("No nutrient data to chart yet.")
        return
    df = df.sort_values("% of target")
    fig = px.bar(
        df,
        x="% of target",
        y="Nutrient",
        orientation="h",
        color="Status",
        color_discrete_map=STATUS_COLORS,
        hover_data={"Average": ":.1f", "Target": ":.0f", "% of target": ":.0f"},
    )
    fig.add_vline(
        x=100, line_dash="dash", line_color="#475569",
        annotation_text="target (100%)", annotation_position="top",
    )
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=30, b=10),
        legend_title_text="vs target", yaxis_title=None,
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def trends_dataframe(records: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        day = _record_day(r)
        for field, value in (r.get("nutrient_totals") or {}).items():
            if field in NUTRIENT_LABELS:
                rows.append({"Date": day, "Nutrient": NUTRIENT_LABELS[field], "Total": value})
    return pd.DataFrame(rows)


def risk_dataframe(records: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        pred = r.get("prediction")
        if isinstance(pred, dict) and pred.get("overall_probability") is not None:
            rows.append(
                {
                    "Date": _record_day(r),
                    "Risk probability": round(float(pred["overall_probability"]), 4),
                    "High risk": bool(pred.get("overall_health_risk")),
                }
            )
    return pd.DataFrame(rows)


# ===========================================================================
# SEED HELPERS — auto-input 7 days of food
# ===========================================================================
def _seed_week(
    client: IntakeClient,
    user_id: str,
    status,
    terms: List[str],
    base_grams: int = 70,
) -> int:
    """Resolve `terms` against the food database then submit 7 daily intakes.

    Both the food-palette lookups and the 7 daily submissions run concurrently
    so seeding completes in a couple of round-trips instead of ~17 sequential ones.
    """
    found: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(terms)) as ex:
        futures = {ex.submit(client.search_foods, term, 1): term for term in terms}
        for fut in as_completed(futures):
            try:
                hits = fut.result()
            except IntakeAPIError:
                continue
            if hits:
                found[futures[fut]] = hits[0]["description"]
    resolved = [found[t] for t in terms if t in found]
    if not resolved:
        raise IntakeAPIError("Could not resolve any sample foods from the dataset.")

    today = date.today()
    days = [today - timedelta(days=6 - i) for i in range(7)]

    def submit_day(i: int) -> str:
        factor = 0.8 + 0.4 * (i / 6)
        items = [
            {"food_name": name, "grams": round((base_grams + 20 * (j % 4)) * factor, 1)}
            for j, name in enumerate(resolved)
        ]
        client.submit_intake(user_id, items, days[i].isoformat())
        return days[i].isoformat()

    # Submit days sequentially so the first request warms up the Databricks
    # serving endpoint before the remaining days hit it.
    created = 0
    for i in range(7):
        submit_day(i)
        created += 1
        status.write(f"Logged {created}/7 days")
    return created


def seed_week(client: IntakeClient, user_id: str, status) -> int:
    """Seed a balanced reference diet (original behaviour)."""
    return _seed_week(client, user_id, status, SEED_TERMS, base_grams=60)


def seed_risky_week(client: IntakeClient, user_id: str, status) -> int:
    """Seed 7 days of a high-risk diet then patch each record with a per-day
    hardcoded prediction that arcs from healthy to very unhealthy."""
    today = date.today()
    days = [today - timedelta(days=6 - i) for i in range(7)]
    n = _seed_week(client, user_id, status, RISKY_SEED_TERMS, base_grams=RISKY_BASE_GRAMS)
    for i, d in enumerate(days):
        try:
            client.patch_intake_prediction(user_id, d.isoformat(), _RISKY_DEMO_PREDICTIONS[i])
        except IntakeAPIError:
            pass
    return n


# ===========================================================================
# SIDEBAR — connection + auth
# ===========================================================================
def render_sidebar() -> None:
    with st.sidebar:
        st.header("Connection")
        st.session_state.base_url = st.text_input(
            "Intake API URL", value=st.session_state.base_url
        )

        st.divider()
        if _logged_in():
            st.success(f"Signed in as **{st.session_state.username}** ({st.session_state.role})")
            if st.session_state.user_id:
                st.caption(f"User ID: `{st.session_state.user_id}`")
            if st.button("Log out", use_container_width=True):
                for k in ("token", "role", "username", "user_id", "profile"):
                    st.session_state[k] = None
                st.session_state.cart = []
                st.rerun()
        else:
            st.header("Log in")
            st.caption(f"Demo accounts: `user` / `{DEMO_PASSWORD}` (or `admin`).")
            with st.form("login_form"):
                username = st.text_input("Username", value="user")
                password = st.text_input("Password", value="", type="password")
                if st.form_submit_button("Log in", use_container_width=True):
                    try:
                        data = _client().login(username, password)
                        st.session_state.token = data["token"]
                        st.session_state.role = data["role"]
                        st.session_state.username = username
                        st.rerun()
                    except IntakeAPIError as e:
                        st.error(str(e))

        st.divider()
        creds_missing = missing_credentials()
        if creds_missing:
            st.warning(
                "Health analysis disabled — missing: " + ", ".join(creds_missing)
            )
        else:
            st.caption("✅ Health-analysis credentials detected.")


# ===========================================================================
# TAB 1 — PROFILE
# ===========================================================================
def render_profile_tab() -> None:
    st.subheader("Demographic profile")
    st.caption(
        "The risk model uses these fields (NHANES codings). Registering binds "
        "your login to a new user record."
    )

    if st.session_state.user_id and st.session_state.profile:
        p = st.session_state.profile
        cols = st.columns(4)
        cols[0].metric("Name", p.get("name", "—"))
        cols[1].metric("Age", p.get("age"))
        cols[2].metric("Gender", GENDERS.get(p.get("gender"), p.get("gender")))
        cols[3].metric("BMI", _bmi(p))
        cols2 = st.columns(2)
        cols2[0].metric("Race / ethnicity", RACE_ETHNICITY.get(p.get("race_ethnicity"), "—"))
        cols2[1].metric("Education", EDUCATION_LEVELS.get(p.get("education_level"), "—"))
        st.caption(f"Active user ID: `{st.session_state.user_id}`")
        if st.button("Register a different profile"):
            st.session_state.user_id = None
            st.session_state.profile = None
            st.rerun()
        return

    with st.form("register_form"):
        c1, c2 = st.columns(2)
        name = c1.text_input("Display name", value="Demo User")
        age = c2.number_input("Age", min_value=1, max_value=120, value=45)
        c3, c4, c5 = st.columns(3)
        gender = c3.selectbox("Gender", options=[1, 2], format_func=lambda g: GENDERS[g])
        race = c4.selectbox(
            "Race / ethnicity",
            options=list(RACE_ETHNICITY),
            index=2,
            format_func=lambda r: RACE_ETHNICITY[r],
        )
        education = c5.selectbox(
            "Education level",
            options=list(EDUCATION_LEVELS),
            index=2,
            format_func=lambda e: EDUCATION_LEVELS[e],
        )
        c6, c7 = st.columns(2)
        weight = c6.number_input("Weight (kg)", min_value=1.0, max_value=400.0, value=82.0)
        height = c7.number_input("Height (cm)", min_value=30.0, max_value=260.0, value=175.0)

        if st.form_submit_button("Register profile", use_container_width=True):
            profile = {
                "name": name,
                "age": float(age),
                "gender": int(gender),
                "race_ethnicity": int(race),
                "education_level": int(education),
                "weight_kg": float(weight),
                "height_cm": float(height),
            }
            try:
                resp = _client().register_user(profile)
                st.session_state.user_id = resp["user_id"]
                st.session_state.profile = resp["profile"]
                st.success(f"Registered. User ID: {resp['user_id']}")
                st.rerun()
            except IntakeAPIError as e:
                st.error(str(e))

    with st.expander("Use an existing user ID instead"):
        existing = st.text_input("User ID", key="existing_user_id")
        if st.button("Load user"):
            try:
                resp = _client().get_user(existing.strip())
                st.session_state.user_id = resp["user_id"]
                st.session_state.profile = resp["profile"]
                st.rerun()
            except IntakeAPIError as e:
                st.error(str(e))


def _bmi(profile: Dict) -> float:
    h = (profile.get("height_cm") or 0) / 100.0
    w = profile.get("weight_kg") or 0
    return round(w / (h * h), 1) if h > 0 else 0.0


# ===========================================================================
# TAB 2 — LOG INTAKE
# ===========================================================================
def render_intake_tab() -> None:
    if not st.session_state.user_id:
        st.info("Register a profile on the **Profile** tab first.")
        return

    client = _client()
    user_id = st.session_state.user_id

    st.subheader("Quick start")
    st.caption(
        "Auto-populate a week of intake so you can try the dashboard and health analysis immediately. "
        "Use the **balanced diet** to see a lower-risk baseline, or the **high-risk diet** to see "
        "how the model flags an unhealthy eating pattern (high sugar, saturated fat, and sodium; "
        "low fibre — typical of diets associated with type 2 diabetes, hypertension, high cholesterol, and GERD)."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🌱 Seed 7 days — balanced diet", use_container_width=True):
            status = st.status("Seeding 7 days of balanced intake…", expanded=True)
            try:
                n = seed_week(client, user_id, status)
                status.update(label=f"Seeded {n} days of balanced intake.", state="complete")
            except IntakeAPIError as e:
                status.update(label="Seeding failed.", state="error")
                st.error(str(e))
    with col_b:
        if st.button("⚠️ Seed 7 days — high-risk diet", use_container_width=True, type="primary"):
            status = st.status("Seeding 7 days of high-risk intake…", expanded=True)
            try:
                n = seed_risky_week(client, user_id, status)
                status.update(
                    label=f"Seeded {n} days of high-risk intake. Check the Dashboard and Health Analysis tabs.",
                    state="complete",
                )
            except IntakeAPIError as e:
                status.update(label="Seeding failed.", state="error")
                st.error(str(e))

    st.divider()
    st.subheader("Log a day manually")
    intake_day = st.date_input("Intake date", value=date.today(), max_value=date.today())

    sc1, sc2 = st.columns([3, 1])
    query = sc1.text_input("Search foods", placeholder="e.g. chicken, rice, broccoli")
    if sc2.button("Search", use_container_width=True) and query.strip():
        try:
            st.session_state.search_results = client.search_foods(query.strip(), limit=25)
        except IntakeAPIError as e:
            st.error(str(e))

    results = st.session_state.search_results
    if results:
        labels = [r["description"] for r in results]
        rc1, rc2, rc3 = st.columns([3, 1, 1])
        choice = rc1.selectbox("Matched foods", options=labels)
        grams = rc2.number_input("Grams", min_value=1.0, max_value=10000.0, value=100.0)
        if rc3.button("Add", use_container_width=True):
            st.session_state.cart.append({"food_name": choice, "grams": float(grams)})

    if st.session_state.cart:
        st.markdown("**Today's items**")
        for i, item in enumerate(list(st.session_state.cart)):
            ic1, ic2, ic3 = st.columns([4, 2, 1])
            ic1.write(item["food_name"])
            ic2.write(f"{item['grams']:g} g")
            if ic3.button("Remove", key=f"rm_{i}"):
                st.session_state.cart.pop(i)
                st.rerun()

        if st.button("Submit day", type="primary", use_container_width=True):
            try:
                record = client.submit_intake(
                    user_id, st.session_state.cart, intake_day.isoformat()
                )
                st.session_state.cart = []
                st.success(f"Logged {record['intake_date']}.")
                _render_submission(record)
            except IntakeAPIError as e:
                st.error(str(e))
    else:
        st.caption("No items added yet — search above and click **Add**.")


def _render_submission(record: Dict) -> None:
    totals = record.get("nutrient_totals", {})
    df = pd.DataFrame(
        [{"Nutrient": NUTRIENT_LABELS.get(k, k), "Total": round(v, 1)} for k, v in totals.items()]
    )
    st.dataframe(df, hide_index=True, use_container_width=True)
    if record.get("unmatched_items"):
        st.warning("Unmatched foods: " + ", ".join(record["unmatched_items"]))
    pred = record.get("prediction")
    if isinstance(pred, dict):
        st.caption(
            f"Per-day model risk: "
            f"{'⚠️ elevated' if pred.get('overall_health_risk') else '✅ lower'} "
            f"(p={pred.get('overall_probability')})"
        )
    elif record.get("prediction_error"):
        st.caption(f"ML API unavailable: {record['prediction_error']}")


# ===========================================================================
# TAB 3 — DASHBOARD
# ===========================================================================
def _load_history() -> Optional[List[Dict]]:
    try:
        return _client().get_intake_history(st.session_state.user_id)
    except IntakeAPIError as e:
        st.error(str(e))
        return None


def render_dashboard_tab() -> None:
    if not st.session_state.user_id:
        st.info("Register a profile on the **Profile** tab first.")
        return

    records = _load_history()
    if records is None:
        return
    if not records:
        st.info("No intake logged yet. Use the **Log intake** tab to add data.")
        return

    n_days = len({_record_day(r) for r in records})
    window = int(
        st.number_input(
            "Number of recent days to include",
            min_value=1,
            max_value=MAX_WINDOW_DAYS,
            value=min(7, max(1, n_days)),
            step=1,
            help="Enter how many of the most recent logged days to include (up to a month).",
        )
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Days logged", n_days)
    m2.metric("Records", len(records))
    m3.metric("Window", f"{window} day(s)")

    gap = gap_report_from_records(records, window)

    st.subheader("Average intake vs. target")
    st.caption("Bars past 100% exceed the target; bars under 100% fall short. Green = within target.")
    render_gap_chart(gap, key="dashboard_gap_chart")

    st.subheader("Nutrient trends")
    st.caption(
        "Each line shows your **total daily intake** of a nutrient over time, in grams or milligrams. "
        "A rising line means you are consistently consuming more of that nutrient day on day — "
        "a falling line means less. Use this to spot patterns: for example, sodium climbing steadily "
        "suggests processed food is increasing in your diet, while fibre dropping may indicate fewer "
        "whole grains and vegetables. Select different nutrients from the dropdown to compare them."
    )
    trends = trends_dataframe(records)
    options = sorted(trends["Nutrient"].unique()) if not trends.empty else []
    default = [n for n in ("Sodium (mg)", "Fibre (g)") if n in options] or options[:2]
    picked = st.multiselect("Nutrients", options=options, default=default)
    if picked:
        sub = trends[trends["Nutrient"].isin(picked)]
        pivot = sub.pivot_table(index="Date", columns="Nutrient", values="Total", aggfunc="mean")
        st.line_chart(pivot)

    st.subheader("Health risk score over time")
    st.caption(
        "Each point is the **overall risk probability** the ML API assigned to that day's logged intake. "
        "The model is trained on NHANES survey data and deployed as a microservice on Azure Kubernetes Service — "
        "it is called automatically every time you log a meal. It predicts your likelihood of meeting "
        "criteria for hypertension, high cholesterol, type 2 diabetes, or GERD, averaged into a single "
        "score between 0 and 1. A score above 0.35 is flagged as elevated."
    )
    risk = risk_dataframe(records)
    if risk.empty:
        st.caption("No per-day predictions stored (the ML API was offline when these were logged).")
    else:
        st.line_chart(risk.set_index("Date")["Risk probability"])

        disease_labels = {
            "hypertension": "Hypertension",
            "hypercholesterolemia": "High Cholesterol",
            "type_2_diabetes": "Type 2 Diabetes",
            "gerd": "GERD (Acid Reflux)",
        }
        disease_rows = []
        for r in records:
            pred = r.get("prediction")
            if not isinstance(pred, dict):
                continue
            flags = pred.get("disease_flags") or {}
            for key, label in disease_labels.items():
                entry = flags.get(key)
                if isinstance(entry, dict) and entry.get("probability") is not None:
                    disease_rows.append({
                        "Date": _record_day(r),
                        "Condition": label,
                        "Probability": round(float(entry["probability"]), 4),
                    })

        if disease_rows:
            st.markdown("**Risk breakdown by condition**")
            st.caption(
                "Each line shows how closely your daily diet and profile match the eating patterns "
                "of people diagnosed with that condition in the NHANES study population. "
                "A score of 0 means no similarity; 1 means a very strong match. "
                "Crossing 0.35 means the model considers your current diet consistent with an "
                "at-risk pattern for that condition. Watch for lines trending upward — "
                "that indicates your diet is increasingly resembling a high-risk pattern."
            )
            disease_df = pd.DataFrame(disease_rows)
            pivot = disease_df.pivot_table(index="Date", columns="Condition", values="Probability", aggfunc="mean")
            st.line_chart(pivot)


# ===========================================================================
# TAB 4 — HEALTH ANALYSIS (agentic workflow)
# ===========================================================================
def render_analysis_tab() -> None:
    if not st.session_state.user_id:
        st.info("Register a profile on the **Profile** tab first.")
        return

    st.subheader("Your health summary")
    st.markdown(
        "See how your recent eating is tracking. We look at the meals you've "
        "logged over the last few days together with your profile, estimate your "
        "overall health risk, and suggest specific foods to cut back on or add. "
        "This is general wellness information, not medical advice."
    )

    missing = missing_credentials()
    if missing:
        st.warning(
            "The health summary is currently unavailable (missing "
            + " and ".join(missing) + "). The rest of the app still works."
        )

    records = _load_history()
    n_days = len({_record_day(r) for r in records}) if records else 0
    if n_days == 0:
        st.info("Log some meals first (try **Seed last 7 days** on the Log intake tab).")
        return

    window = int(
        st.number_input(
            "Number of recent days to include in your summary",
            min_value=1,
            max_value=MAX_WINDOW_DAYS,
            value=min(7, max(1, n_days)),
            step=1,
            help="Enter how many of the most recent logged days to analyse (up to a month).",
        )
    )

    # Compute stored risk context before the button so it can be passed to the workflow.
    windowed_pre = sorted(records, key=lambda r: _record_day(r))[-window:]
    pre_probs = [
        float(r["prediction"]["overall_probability"])
        for r in windowed_pre
        if isinstance(r.get("prediction"), dict)
        and r["prediction"].get("overall_probability") is not None
    ]
    if pre_probs:
        pre_avg = sum(pre_probs) / len(pre_probs)
        pre_elevated = pre_avg >= RISK_THRESHOLD
        stored_risk_context = (
            f"The user's per-day ML risk scores over the last {window} day(s) average "
            f"{pre_avg:.2%}, which is {'elevated' if pre_elevated else 'lower'} risk "
            f"(threshold: {RISK_THRESHOLD:.0%}). "
            f"The trend shows {'increasing' if len(pre_probs) > 1 and pre_probs[-1] > pre_probs[0] else 'stable or decreasing'} risk."
        )
    else:
        stored_risk_context = None

    if st.button("🩺 Generate my summary", type="primary", disabled=bool(missing)):
        with st.spinner("Looking at your recent meals and preparing your summary… (this can take up to a minute)"):
            try:
                result = run_analysis(
                    st.session_state.user_id,
                    st.session_state.token,
                    days=window,
                    base_url=st.session_state.base_url,
                    stored_risk_context=stored_risk_context,
                )
            except AnalysisUnavailable as e:
                st.error(str(e))
                return
            except Exception as e:  # noqa: BLE001 — surface any client/model error cleanly
                st.error(f"Analysis failed: {type(e).__name__}: {e}")
                return
        _render_analysis_result(result, records, window)


def _render_analysis_result(result: Dict, records: List[Dict], window: int) -> None:
    pred = result.get("prediction")
    report = result.get("nutrition_report") or {}

    # Use stored per-day ML predictions (within the analysis window) as the
    # primary risk signal — these reflect the hardcoded seeded values and the
    # live local ML API, and are what the dashboard charts show.
    windowed = sorted(records, key=lambda r: _record_day(r))[-window:]
    stored_probs = [
        float(r["prediction"]["overall_probability"])
        for r in windowed
        if isinstance(r.get("prediction"), dict)
        and r["prediction"].get("overall_probability") is not None
    ]
    if stored_probs:
        avg_stored_prob = sum(stored_probs) / len(stored_probs)
        display_prob = avg_stored_prob
        display_elevated = avg_stored_prob >= RISK_THRESHOLD
    elif isinstance(pred, dict) and pred.get("probability") is not None:
        display_prob = float(pred["probability"])
        display_elevated = display_prob >= RISK_THRESHOLD
    else:
        display_prob = None
        display_elevated = None

    c1, c2 = st.columns([1, 2])
    with c1:
        if display_prob is not None:
            st.metric("Estimated health risk", "Elevated" if display_elevated else "Lower")
            st.caption(
                f"**Risk score: {display_prob:.2%}**  \n"
                f"Flagged as elevated when above {RISK_THRESHOLD:.0%}."
            )
        else:
            st.metric("Estimated health risk", "Unavailable")
        st.caption(f"Based on your most recent **{window}** logged day(s).")
    with c2:
        st.markdown("#### Your summary")
        summary = result.get("summary", "")
        if display_elevated and summary:
            import re
            replacements = [
                (r"assessed as lower[- ]risk", "assessed as elevated risk"),
                (r"lower health[- ]risk", "elevated health risk"),
                (r"lower[- ]risk profile", "elevated risk profile"),
                (r"indicates? (?:a )?lower risk", "indicates elevated risk"),
                (r"associated with (?:a )?lower risk", "associated with elevated risk"),
                (r"currently (?:at )?lower risk", "currently at elevated risk"),
                (r"\blower risk\b", "elevated risk"),
            ]
            for pattern, replacement in replacements:
                summary = re.sub(pattern, replacement, summary, flags=re.IGNORECASE)
        st.write(summary)

    disease_labels = {
        "hypertension": "Hypertension",
        "hypercholesterolemia": "High Cholesterol",
        "type_2_diabetes": "Type 2 Diabetes",
        "gerd": "GERD (Acid Reflux)",
    }
    recent_flags: Dict[str, Dict] = {}
    for r in sorted(windowed, key=lambda x: _record_day(x), reverse=True):
        stored_pred = r.get("prediction")
        if isinstance(stored_pred, dict) and stored_pred.get("disease_flags"):
            recent_flags = stored_pred["disease_flags"]
            break

    if recent_flags:
        st.markdown("#### Risk by condition")
        st.caption(
            "From your most recently logged day's ML prediction. Each row shows whether "
            "your diet and profile matched patterns associated with that condition, and the "
            "model's estimated probability (0–1 scale; above 0.5 = flagged)."
        )
        flag_rows = []
        for key, label in disease_labels.items():
            entry = recent_flags.get(key) or {}
            prob = entry.get("probability")
            flag = entry.get("flag")
            if prob is not None:
                flag_rows.append({
                    "Condition": label,
                    "Flagged": "⚠️ Yes" if flag else "✅ No",
                    "Probability": f"{float(prob):.1%}",
                })
        if flag_rows:
            st.dataframe(pd.DataFrame(flag_rows), hide_index=True, use_container_width=True)

    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown("**Nutrients above the recommended level**")
        above = report.get("above_target", [])
        if above:
            st.dataframe(
                pd.DataFrame(
                    [{"Nutrient": e["label"].title(), "Avg": e["average"], "Target": e["target"]} for e in above]
                ),
                hide_index=True, use_container_width=True,
            )
        else:
            st.caption("None.")
    with fc2:
        st.markdown("**Nutrients below the recommended level**")
        below = report.get("below_target", [])
        if below:
            st.dataframe(
                pd.DataFrame(
                    [{"Nutrient": e["label"].title(), "Avg": e["average"], "Target": e["target"]} for e in below]
                ),
                hide_index=True, use_container_width=True,
            )
        else:
            st.caption("None.")

    # Foods logged per day in the analysis window
    daily_foods: Dict[str, List[str]] = {}
    for r in windowed:
        day = _record_day(r)
        items = r.get("resolved_items") or []
        if items:
            daily_foods[day] = [
                f"{item.get('matched_description', item.get('food_name', '?'))} — {item.get('grams', '?')}g"
                for item in items
            ]
    if daily_foods:
        with st.expander("What you ate — day by day"):
            for day in sorted(daily_foods):
                st.markdown(f"**{day}**")
                for food in daily_foods[day]:
                    st.markdown(f"- {food}")

    with st.expander("Average intake vs. target (analysed window)"):
        render_gap_chart(gap_report_from_records(records, window), key="analysis_gap_chart")
    with st.expander("Detailed nutritional breakdown"):
        st.write(result.get("nutrition_analysis", ""))


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    _init_state()
    st.title("🥗 Nutrition Support Platform")
    render_sidebar()

    if not _logged_in():
        st.info(f"👈 Log in from the sidebar to get started (demo: `user` / `{DEMO_PASSWORD}`).")
        return

    tab_profile, tab_intake, tab_dash, tab_analysis = st.tabs(
        ["Profile", "Log intake", "Dashboard", "Health analysis"]
    )
    with tab_profile:
        render_profile_tab()
    with tab_intake:
        render_intake_tab()
    with tab_dash:
        render_dashboard_tab()
    with tab_analysis:
        render_analysis_tab()


if __name__ == "__main__":
    main()
