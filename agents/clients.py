"""
clients.py — HTTP clients used by the agentic workflow.

Two clients:

  IntakeAPIClient
      Talks to Microservice 1 (the Intake API). Handles JWT login and the
      read endpoints the agents need: user profile and intake history. This is
      the *only* place that knows the Intake API's URL shape and auth flow.

  DatabricksRiskClient
      Talks to the new ML model, now hosted on Databricks and served through
      MLflow as a REST endpoint. Replaces the old in-cluster `ml-api` service.
      Sends the standard MLflow scoring payload and normalises the response to
      a simple {label, probability} shape regardless of how the registered
      model signature spells its output.

Both clients are thin, synchronous wrappers over httpx with explicit timeouts
and clear error messages — no hidden retries, so the orchestrator stays in
control of failure handling.
"""

import logging
import os
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger("agents.clients")


# ===========================================================================
# INTAKE API CLIENT — Microservice 1
# ===========================================================================
class IntakeAPIError(RuntimeError):
    """Raised when the Intake API returns a non-success response."""


class IntakeAPIClient:
    """Minimal client for the Intake API used by both agents.

    The base URL is whatever address actually fronts the service:
      * direct / local run:  http://localhost:8001
      * through the AKS ingress: https://<host>/api   (the ingress strips
        the `/api` prefix before the request reaches the service, so paths
        are appended bare in both cases).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get(
            "INTAKE_API_URL", "http://localhost:8001"
        )).rstrip("/")
        self.token = token
        self.timeout = timeout

    # ---- auth ------------------------------------------------------------
    def login(self, username: str, password: str) -> str:
        """Authenticate and cache the JWT for subsequent calls. Returns the token."""
        resp = httpx.post(
            f"{self.base_url}/v1/login",
            json={"username": username, "password": password},
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise IntakeAPIError(
                f"Login failed ({resp.status_code}): {resp.text[:200]}"
            )
        self.token = resp.json()["token"]
        logger.info("intake_login_ok user=%s base=%s", username, self.base_url)
        return self.token

    def _headers(self) -> Dict[str, str]:
        if not self.token:
            raise IntakeAPIError("No token set — call login() first or pass token=")
        return {"Authorization": f"Bearer {self.token}"}

    # ---- reads used by the agents ---------------------------------------
    def get_user(self, user_id: str) -> Dict:
        """GET /users/{user_id} — demographic profile."""
        resp = httpx.get(
            f"{self.base_url}/users/{user_id}",
            headers=self._headers(),
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise IntakeAPIError(
                f"get_user({user_id}) failed ({resp.status_code}): {resp.text[:200]}"
            )
        return resp.json()

    def get_intake_history(self, user_id: str) -> List[Dict]:
        """GET /users/{user_id}/intake — every stored intake record, newest last."""
        resp = httpx.get(
            f"{self.base_url}/users/{user_id}/intake",
            headers=self._headers(),
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise IntakeAPIError(
                f"get_intake_history({user_id}) failed "
                f"({resp.status_code}): {resp.text[:200]}"
            )
        return resp.json().get("records", [])


# ===========================================================================
# DATABRICKS RISK CLIENT — the MLflow REST serving endpoint
# ===========================================================================
class DatabricksRiskError(RuntimeError):
    """Raised when the Databricks serving endpoint errors or returns nothing usable."""


class DatabricksRiskClient:
    """Client for the `health_food_risk_detector` model served on Databricks.

    The endpoint is a standard MLflow model-serving `/invocations` URL. It is
    authenticated with a Databricks personal-access (or service-principal)
    token sent as a Bearer credential. We POST the MLflow `dataframe_records`
    payload — the same shape the ml-api service sends to this endpoint (see
    ml_api.py) — and read back the `predictions` array.

    The registered model is a Spark ML pipeline whose logged signature outputs
    the `prediction` column — a single binary label (`high_health_risk_label`,
    1 = elevated risk), with no probability. `_parse` is tolerant of however
    MLflow serialises that (bare scalar, one-element list, or a dict),
    flattening all three into {label, probability, raw}.
    """

    DEFAULT_ENDPOINT = (
        "https://<your-workspace>.cloud.databricks.com"
        "/serving-endpoints/health-food-risk-endpoint/invocations"
    )

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        token: Optional[str] = None,
        # The serverless serving endpoint scales to zero; a cold start can take
        # over a minute, so allow the same 120s the ml-api service uses.
        timeout: float = 120.0,
    ) -> None:
        self.endpoint_url = endpoint_url or os.environ.get(
            "DATABRICKS_ENDPOINT_URL", self.DEFAULT_ENDPOINT
        )
        self.token = token or os.environ.get("DATABRICKS_TOKEN")
        self.timeout = timeout
        if not self.token:
            raise DatabricksRiskError(
                "No Databricks token. Set DATABRICKS_TOKEN (a personal-access or "
                "service-principal token) or pass token= to DatabricksRiskClient."
            )

    def predict(self, features: Dict[str, Optional[float]]) -> Dict:
        """Score one feature record. Returns {label, probability, raw}.

        `features` is a single {column: value} mapping covering the model's full
        106-column signature (null for columns to be imputed) — see
        risk_agent.build_feature_record(), the one place that assembles it. It is
        sent as one MLflow `dataframe_records` row, matching the ml-api service.
        """
        payload = {"dataframe_records": [features]}
        try:
            resp = httpx.post(
                self.endpoint_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        except httpx.RequestError as e:
            raise DatabricksRiskError(f"Databricks endpoint unreachable: {e}") from e

        if resp.status_code != 200:
            raise DatabricksRiskError(
                f"Databricks endpoint returned {resp.status_code}: {resp.text[:300]}"
            )
        return self._parse(resp.json())

    @staticmethod
    def _parse(body: Dict) -> Dict:
        """Normalise MLflow's response into {label, probability, raw}."""
        preds = body.get("predictions", body)
        if isinstance(preds, list):
            if not preds:
                raise DatabricksRiskError(f"Empty predictions in response: {body}")
            first = preds[0]
        else:
            first = preds

        label: Optional[int] = None
        probability: Optional[float] = None

        if isinstance(first, dict):
            # e.g. {"high_health_risk_label": 1, "probability": 0.83}
            for k, v in first.items():
                key = k.lower()
                if "prob" in key or "score" in key:
                    probability = float(v)
                elif "label" in key or "predict" in key or "risk" in key:
                    label = int(round(float(v)))
            if label is None and probability is not None:
                label = int(probability >= 0.5)
        else:
            # bare scalar — the label (0/1) or a probability in [0, 1]
            value = float(first)
            if 0.0 <= value <= 1.0 and value not in (0.0, 1.0):
                probability = value
                label = int(value >= 0.5)
            else:
                label = int(round(value))

        if label is None:
            raise DatabricksRiskError(f"Could not read a label from response: {body}")
        return {"label": label, "probability": probability, "raw": body}
