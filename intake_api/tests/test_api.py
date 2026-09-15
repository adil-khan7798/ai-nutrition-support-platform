"""API tests for the Intake API, driven through FastAPI's TestClient.

The outbound call to the ML API is stubbed with respx so these tests never need
a live ML service. Every test uses the `client` fixture, which loads the
fixture food dataset via the app lifespan and gives each test a fresh store.
"""

import datetime

import httpx
import respx

ML_PREDICT_URL = "http://ml-api.test/predict"

# A canned ML response matching the ml_api PredictResponse schema.
FAKE_PREDICTION = {
    "overall_health_risk": True,
    "overall_probability": 0.62,
    "disease_flags": {
        "hypertension": {"flag": True, "probability": 0.71},
        "hypercholesterolemia": {"flag": False, "probability": 0.30},
        "type_2_diabetes": {"flag": True, "probability": 0.55},
        "gerd": {"flag": False, "probability": 0.10},
    },
}


def register(client, profile, auth_headers):
    """Helper: register a user and return their user_id."""
    resp = client.post("/users", json=profile, headers=auth_headers)
    assert resp.status_code == 201
    return resp.json()["user_id"]


# --------------------------------------------------------------------------
# health / banner
# --------------------------------------------------------------------------
def test_root_banner(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["service"] == "intake-api"


def test_health_reports_foods_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["foods_loaded"] == 2  # two rows in the fixture CSV


# --------------------------------------------------------------------------
# user registration / retrieval
# --------------------------------------------------------------------------
def test_register_user_returns_201_and_id(client, valid_profile, auth_headers):
    resp = client.post("/users", json=valid_profile, headers=auth_headers)
    assert resp.status_code == 201
    body = resp.json()
    assert body["user_id"]
    assert body["profile"]["name"] == "Test User"


def test_get_user_roundtrip(client, valid_profile, auth_headers):
    uid = register(client, valid_profile, auth_headers)
    resp = client.get(f"/users/{uid}", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["user_id"] == uid


def test_get_unknown_user_404(client, auth_headers):
    resp = client.get("/users/does-not-exist", headers=auth_headers)
    assert resp.status_code == 404


def test_register_rejects_invalid_gender(client, valid_profile, auth_headers):
    bad = {**valid_profile, "gender": 3}  # constraint is 1..2
    resp = client.post("/users", json=bad, headers=auth_headers)
    assert resp.status_code == 422


def test_register_rejects_out_of_range_age(client, valid_profile, auth_headers):
    bad = {**valid_profile, "age": 200}  # constraint is 0..120
    resp = client.post("/users", json=bad, headers=auth_headers)
    assert resp.status_code == 422


# --------------------------------------------------------------------------
# food search
# --------------------------------------------------------------------------
def test_food_search_returns_matches(client, auth_headers):
    resp = client.get("/foods/search", params={"q": "apple"}, headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["results"][0]["description"] == "Apple raw"


def test_food_search_requires_query(client, auth_headers):
    resp = client.get("/foods/search", params={"q": ""}, headers=auth_headers)
    assert resp.status_code == 422  # min_length=1 on q


# --------------------------------------------------------------------------
# intake submission (ML call stubbed)
# --------------------------------------------------------------------------
@respx.mock
def test_submit_intake_embeds_prediction(client, valid_profile, auth_headers):
    respx.post(ML_PREDICT_URL).mock(
        return_value=httpx.Response(200, json=FAKE_PREDICTION)
    )
    uid = register(client, valid_profile, auth_headers)
    resp = client.post(
        f"/users/{uid}/intake",
        json={"items": [{"food_name": "Apple raw", "grams": 200}]},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] == FAKE_PREDICTION
    assert body["prediction_error"] is None
    # 200g apple -> protein 0.6, sugar 20.0
    assert body["nutrient_totals"]["protein_g"] == 0.6
    assert body["nutrient_totals"]["sugar_g"] == 20.0


@respx.mock
def test_submit_intake_forwards_profile_and_totals_to_ml(
    client, valid_profile, auth_headers
):
    route = respx.post(ML_PREDICT_URL).mock(
        return_value=httpx.Response(200, json=FAKE_PREDICTION)
    )
    uid = register(client, valid_profile, auth_headers)
    client.post(
        f"/users/{uid}/intake",
        json={"items": [{"food_name": "Apple raw", "grams": 100}]},
        headers=auth_headers,
    )
    assert route.called
    sent = route.calls.last.request
    import json

    payload = json.loads(sent.content)
    # demographics from the profile are forwarded
    assert payload["age"] == valid_profile["age"]
    assert payload["gender"] == valid_profile["gender"]
    # aggregated nutrient totals are merged in
    assert payload["protein_g"] == 0.3
    assert payload["sodium_mg"] == 1.0


@respx.mock
def test_submit_intake_predicts_from_available_daily_average(
    client, valid_profile, auth_headers
):
    route = respx.post(ML_PREDICT_URL).mock(
        return_value=httpx.Response(200, json=FAKE_PREDICTION)
    )
    uid = register(client, valid_profile, auth_headers)
    client.post(
        f"/users/{uid}/intake",
        json={"items": [{"food_name": "Apple raw", "grams": 100}]},
        headers=auth_headers,
    )
    resp = client.post(
        f"/users/{uid}/intake",
        json={"items": [{"food_name": "Beef patty cooked", "grams": 100}]},
        headers=auth_headers,
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction_window_days"] == 2
    assert body["nutrient_totals"]["protein_g"] == 26.0
    assert body["weekly_average_nutrient_totals"]["protein_g"] == 13.15
    assert body["weekly_average_nutrient_totals"]["sodium_mg"] == 38.0

    sent = route.calls.last.request
    import json

    payload = json.loads(sent.content)
    assert payload["protein_g"] == 13.15
    assert payload["sodium_mg"] == 38.0


@respx.mock
def test_submit_intake_prediction_average_uses_latest_seven_days(
    client, valid_profile, auth_headers
):
    route = respx.post(ML_PREDICT_URL).mock(
        return_value=httpx.Response(200, json=FAKE_PREDICTION)
    )
    uid = register(client, valid_profile, auth_headers)
    for day in range(1, 9):
        resp = client.post(
            f"/users/{uid}/intake",
            json={"items": [{"food_name": "Apple raw", "grams": day * 100}]},
            headers=auth_headers,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction_window_days"] == 7
    assert body["nutrient_totals"]["protein_g"] == 2.4
    assert body["weekly_average_nutrient_totals"]["protein_g"] == 1.5
    assert body["weekly_average_nutrient_totals"]["sugar_g"] == 50.0

    sent = route.calls.last.request
    import json

    payload = json.loads(sent.content)
    assert payload["protein_g"] == 1.5
    assert payload["sugar_g"] == 50.0


@respx.mock
def test_submit_intake_accepts_backfilled_previous_day(
    client, valid_profile, auth_headers
):
    route = respx.post(ML_PREDICT_URL).mock(
        return_value=httpx.Response(200, json=FAKE_PREDICTION)
    )
    uid = register(client, valid_profile, auth_headers)
    today = datetime.datetime.now(datetime.timezone.utc).date()
    day_1 = (today - datetime.timedelta(days=3)).isoformat()
    day_2 = (today - datetime.timedelta(days=2)).isoformat()
    day_3 = (today - datetime.timedelta(days=1)).isoformat()

    client.post(
        f"/users/{uid}/intake",
        json={
            "intake_date": day_1,
            "items": [{"food_name": "Apple raw", "grams": 100}],
        },
        headers=auth_headers,
    )
    client.post(
        f"/users/{uid}/intake",
        json={
            "intake_date": day_3,
            "items": [{"food_name": "Beef patty cooked", "grams": 100}],
        },
        headers=auth_headers,
    )
    resp = client.post(
        f"/users/{uid}/intake",
        json={
            "intake_date": day_2,
            "items": [{"food_name": "Apple raw", "grams": 300}],
        },
        headers=auth_headers,
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["intake_date"] == day_2
    assert body["prediction_window_days"] == 3
    assert body["nutrient_totals"]["protein_g"] == 0.9
    assert body["weekly_average_nutrient_totals"]["protein_g"] == 9.0667
    assert body["weekly_average_nutrient_totals"]["sugar_g"] == 13.3333

    sent = route.calls.last.request
    import json

    payload = json.loads(sent.content)
    assert payload["protein_g"] == 9.0667
    assert payload["sugar_g"] == 13.3333


def test_submit_intake_rejects_future_date(client, valid_profile, auth_headers):
    uid = register(client, valid_profile, auth_headers)
    tomorrow = (
        datetime.datetime.now(datetime.timezone.utc).date() + datetime.timedelta(days=1)
    ).isoformat()
    resp = client.post(
        f"/users/{uid}/intake",
        json={
            "intake_date": tomorrow,
            "items": [{"food_name": "Apple raw", "grams": 100}],
        },
        headers=auth_headers,
    )

    assert resp.status_code == 422


@respx.mock
def test_submit_intake_saves_record_when_ml_unreachable(
    client, valid_profile, auth_headers
):
    """ML failure must not lose the intake — record saves with an error note."""
    respx.post(ML_PREDICT_URL).mock(side_effect=httpx.ConnectError("boom"))
    uid = register(client, valid_profile, auth_headers)
    resp = client.post(
        f"/users/{uid}/intake",
        json={"items": [{"food_name": "Apple raw", "grams": 100}]},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] is None
    assert "unreachable" in body["prediction_error"].lower()


@respx.mock
def test_submit_intake_handles_ml_5xx(client, valid_profile, auth_headers):
    respx.post(ML_PREDICT_URL).mock(
        return_value=httpx.Response(500, text="model exploded")
    )
    uid = register(client, valid_profile, auth_headers)
    resp = client.post(
        f"/users/{uid}/intake",
        json={"items": [{"food_name": "Apple raw", "grams": 100}]},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] is None
    assert "500" in body["prediction_error"]


def test_submit_intake_all_unmatched_returns_422(client, valid_profile, auth_headers):
    """No food matched -> 422, and the ML API is never called."""
    uid = register(client, valid_profile, auth_headers)
    resp = client.post(
        f"/users/{uid}/intake",
        json={"items": [{"food_name": "zzz nonexistent", "grams": 100}]},
        headers=auth_headers,
    )
    assert resp.status_code == 422


def test_submit_intake_unknown_user_404(client, auth_headers):
    resp = client.post(
        "/users/nobody/intake",
        json={"items": [{"food_name": "Apple raw", "grams": 100}]},
        headers=auth_headers,
    )
    assert resp.status_code == 404


def test_submit_intake_rejects_empty_items(client, valid_profile, auth_headers):
    uid = register(client, valid_profile, auth_headers)
    resp = client.post(f"/users/{uid}/intake", json={"items": []}, headers=auth_headers)
    assert resp.status_code == 422  # min_length=1 on items


# --------------------------------------------------------------------------
# intake history
# --------------------------------------------------------------------------
@respx.mock
def test_intake_history_accumulates(client, valid_profile, auth_headers):
    respx.post(ML_PREDICT_URL).mock(
        return_value=httpx.Response(200, json=FAKE_PREDICTION)
    )
    uid = register(client, valid_profile, auth_headers)
    for _ in range(2):
        client.post(
            f"/users/{uid}/intake",
            json={"items": [{"food_name": "Apple raw", "grams": 100}]},
            headers=auth_headers,
        )
    resp = client.get(f"/users/{uid}/intake", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert len(body["records"]) == 2


def test_intake_history_unknown_user_404(client, auth_headers):
    resp = client.get("/users/nobody/intake", headers=auth_headers)
    assert resp.status_code == 404
