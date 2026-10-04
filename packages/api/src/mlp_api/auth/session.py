import time
from datetime import UTC, datetime

import jwt
from fastapi import Request, Response
from fastapi.responses import JSONResponse, RedirectResponse

from mlp_core import api_paths
from mlp_core.config import (
    FAILED_LOGIN_WINDOW,
    JWT_ALGORITHM,
    MAX_FAILED_LOGINS,
    PUBLIC_PATHS,
    SECRET_MAX_AGE,
    SESSION_COOKIE,
    STEP_TOKEN_CLAIM,
    TOKEN_LIFETIME,
    WEB_UI_LOGIN_URL,
)


def too_many_failed_logins(failed_logins: list[float]) -> bool:
    cutoff = time.monotonic() - FAILED_LOGIN_WINDOW.total_seconds()
    failed_logins[:] = [failed_at for failed_at in failed_logins if failed_at > cutoff]
    return len(failed_logins) >= MAX_FAILED_LOGINS


def record_failed_login(failed_logins: list[float]) -> None:
    failed_logins.append(time.monotonic())


def issue_token(jwt_secret: str) -> str:
    return jwt.encode({"exp": datetime.now(UTC) + TOKEN_LIFETIME}, jwt_secret, JWT_ALGORITHM)


# The Pipeline's Secret holds it, so it needs to last only as long as Secrets are kept.
def issue_step_token(jwt_secret: str, pipeline_id: int) -> str:
    claims = {"exp": datetime.now(UTC) + SECRET_MAX_AGE, STEP_TOKEN_CLAIM: pipeline_id}
    return jwt.encode(claims, jwt_secret, JWT_ALGORITHM)


def is_logged_in(request: Request) -> bool:
    claims = token_claims(request)
    return claims is not None and STEP_TOKEN_CLAIM not in claims


# A step token reaches only its own Pipeline's step routes.
def is_pipeline_step(request: Request) -> bool:
    claims = token_claims(request)
    if claims is None or STEP_TOKEN_CLAIM not in claims:
        return False
    paths = [path.format(id=claims[STEP_TOKEN_CLAIM]) for path in api_paths.STEP_PATHS]
    return request.url.path in paths


def token_claims(request: Request) -> dict | None:
    """The claims of the request's bearer token or login cookie, or None if it has no valid one."""
    authorization = request.headers.get("authorization", "")
    if authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ")
    else:
        token = request.cookies.get(SESSION_COOKIE, "")
    try:
        return jwt.decode(
            token, request.app.state.jwt_secret, [JWT_ALGORITHM], options={"require": ["exp"]}
        )
    except jwt.InvalidTokenError:
        return None


async def require_login(request: Request, call_next) -> Response:
    allowed = request.url.path in PUBLIC_PATHS or is_logged_in(request) or is_pipeline_step(request)
    if allowed:
        return await call_next(request)
    # Browsers go to the Web UI's login, also via Traefik's forwardAuth for the KFP and MLflow UIs.
    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(WEB_UI_LOGIN_URL, status_code=303)
    return JSONResponse({"detail": "Not logged in"}, status_code=401)
