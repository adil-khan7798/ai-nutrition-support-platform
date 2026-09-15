"""intake_client.py — a thin httpx client over the Intake API for the frontend.

The agents package ships a read-only `IntakeAPIClient`; the Streamlit app also
needs the write paths (register, submit intake) and food search, so it has its
own small client rather than widening the agents' one. Paths are appended bare
to the base URL — the same shape the agents client uses — which works both for a
direct local run (http://localhost:8001) and behind the AKS ingress (.../api).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import httpx

DEFAULT_BASE_URL = os.environ.get("INTAKE_API_URL", "http://localhost:8001")


class IntakeAPIError(RuntimeError):
    """Raised when the Intake API returns a non-success response."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class IntakeClient:
    """Covers every Intake API endpoint the Streamlit UI touches."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: float = 160.0,
    ) -> None:
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.token = token
        self.timeout = timeout

    # ---- helpers ---------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _raise(self, resp: httpx.Response, action: str) -> None:
        try:
            body = resp.json()
            detail = body.get("message") or body.get("detail") or resp.text
        except Exception:  # noqa: BLE001 — non-JSON error body
            detail = resp.text
        raise IntakeAPIError(
            f"{action} failed ({resp.status_code}): {detail}", resp.status_code
        )

    # ---- auth ------------------------------------------------------------
    def login(self, username: str, password: str) -> Dict:
        """POST /v1/login. Caches the JWT and returns {token, role, expires_in_minutes}."""
        resp = httpx.post(
            f"{self.base_url}/v1/login",
            json={"username": username, "password": password},
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            self._raise(resp, "Login")
        data = resp.json()
        self.token = data["token"]
        return data

    # ---- users -----------------------------------------------------------
    def register_user(self, profile: Dict) -> Dict:
        """POST /users — create a profile. Returns {user_id, profile}."""
        resp = httpx.post(
            f"{self.base_url}/users",
            json=profile,
            headers=self._headers(),
            timeout=self.timeout,
        )
        if resp.status_code not in (200, 201):
            self._raise(resp, "Register user")
        return resp.json()

    def get_user(self, user_id: str) -> Dict:
        """GET /users/{user_id}."""
        resp = httpx.get(
            f"{self.base_url}/users/{user_id}",
            headers=self._headers(),
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            self._raise(resp, "Get user")
        return resp.json()

    # ---- food + intake ---------------------------------------------------
    def search_foods(self, query: str, limit: int = 20) -> List[Dict]:
        """GET /foods/search?q=. Returns the results list."""
        resp = httpx.get(
            f"{self.base_url}/foods/search",
            params={"q": query, "limit": limit},
            headers=self._headers(),
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            self._raise(resp, "Food search")
        return resp.json().get("results", [])

    def submit_intake(
        self,
        user_id: str,
        items: List[Dict],
        intake_date: Optional[str] = None,
    ) -> Dict:
        """POST /users/{user_id}/intake.

        `items` is a list of {"food_name", "grams"}; `intake_date` is an optional
        ISO date string (the API defaults it to today's UTC date).
        """
        payload: Dict = {"items": items}
        if intake_date:
            payload["intake_date"] = intake_date
        resp = httpx.post(
            f"{self.base_url}/users/{user_id}/intake",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            self._raise(resp, "Submit intake")
        return resp.json()

    def get_intake_history(self, user_id: str) -> List[Dict]:
        """GET /users/{user_id}/intake. Returns every stored record, newest last."""
        resp = httpx.get(
            f"{self.base_url}/users/{user_id}/intake",
            headers=self._headers(),
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            self._raise(resp, "Intake history")
        return resp.json().get("records", [])

    def patch_intake_prediction(self, user_id: str, date: str, prediction: Dict) -> None:
        """PATCH /users/{user_id}/intake/{date} — overwrite the stored ML prediction."""
        resp = httpx.patch(
            f"{self.base_url}/users/{user_id}/intake/{date}",
            json=prediction,
            headers=self._headers(),
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            self._raise(resp, "Patch intake prediction")
