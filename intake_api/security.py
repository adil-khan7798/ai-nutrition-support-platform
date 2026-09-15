"""
security.py — JWT authentication and RBAC for FastAPI services.

Environment variables consumed:
  JWT_SECRET_KEY   HMAC signing secret (set via Kubernetes Secret in prod)
  JWT_ALGORITHM    default HS256
  JWT_EXP_MINUTES  token lifetime in minutes, default 60
"""

import os
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

JWT_SECRET: str = os.getenv("JWT_SECRET_KEY", "dev-secret-change-in-production")
JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXP_MINUTES: int = int(os.getenv("JWT_EXP_MINUTES", "60"))

_bearer = HTTPBearer(auto_error=False)


def generate_token(username: str, role: str, exp_minutes: int | None = None) -> str:
    """Return a signed JWT containing sub, role, iat, and exp claims."""
    minutes = exp_minutes if exp_minutes is not None else JWT_EXP_MINUTES
    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "role": role,
        "iat": now,
        "exp": now + timedelta(minutes=minutes),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def peek_token(token: str) -> dict | None:
    """Decode token for logging purposes; returns None instead of raising."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None


def _decode(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "Unauthorized",
                "message": "Token has expired",
                "status_code": 401,
            },
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "Unauthorized",
                "message": "Token is missing, invalid, or expired",
                "status_code": 401,
            },
        )


def token_required(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> dict:
    """FastAPI dependency — validates Bearer token and returns decoded claims."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "Unauthorized",
                "message": "Token is missing, invalid, or expired",
                "status_code": 401,
            },
        )
    return _decode(credentials.credentials)


def role_required(*allowed_roles: str):
    """Return a FastAPI dependency that enforces role membership."""

    def _dep(payload: dict = Depends(token_required)) -> dict:
        if payload.get("role") not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "Forbidden",
                    "message": f"Access requires role: {' or '.join(allowed_roles)}",
                    "status_code": 403,
                },
            )
        return payload

    return _dep
