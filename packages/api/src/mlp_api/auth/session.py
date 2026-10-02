import time
from datetime import UTC, datetime

import jwt
from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse

from mlp_api.auth.account import password_matches
from mlp_core import api_paths
from mlp_core.config import (
    FAILED_LOGIN_WINDOW,
    JWT_ALGORITHM,
    MAX_FAILED_LOGINS,
    PUBLIC_PATHS,
    SESSION_COOKIE,
    TOKEN_LIFETIME,
)


def log_in(request: Request, password: str) -> str:
    """A session token for the shared password; HTTPException with the reason otherwise."""
    failed_logins = request.app.state.failed_logins[request.client.host]
    if too_many_failed_logins(failed_logins):
        raise HTTPException(429, "Too many failed logins, try again later")
    if not password_matches(request.app.state.engine, password):
        record_failed_login(failed_logins)
        raise HTTPException(401, "Wrong password")
    return issue_token(request.app.state.jwt_secret)


def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=int(TOKEN_LIFETIME.total_seconds()),
        httponly=True,
        secure=True,
        samesite="lax",
    )


def too_many_failed_logins(failed_logins: list[float]) -> bool:
    cutoff = time.monotonic() - FAILED_LOGIN_WINDOW.total_seconds()
    failed_logins[:] = [failed_at for failed_at in failed_logins if failed_at > cutoff]
    return len(failed_logins) >= MAX_FAILED_LOGINS


def record_failed_login(failed_logins: list[float]) -> None:
    failed_logins.append(time.monotonic())


def issue_token(jwt_secret: str) -> str:
    return jwt.encode({"exp": datetime.now(UTC) + TOKEN_LIFETIME}, jwt_secret, JWT_ALGORITHM)


def is_logged_in(request: Request) -> bool:
    authorization = request.headers.get("authorization", "")
    if authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ")
    else:
        token = request.cookies.get(SESSION_COOKIE, "")
    try:
        jwt.decode(
            token, request.app.state.jwt_secret, [JWT_ALGORITHM], options={"require": ["exp"]}
        )
    except jwt.InvalidTokenError:
        return False
    return True


async def require_login(request: Request, call_next) -> Response:
    if request.url.path in PUBLIC_PATHS or is_logged_in(request):
        return await call_next(request)
    # Browsers go to the login page, also when Traefik's forwardAuth asks for the KFP or MLflow UI.
    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(api_paths.WEB_LOGIN, status_code=303)
    # htmx follows HX-Redirect, so a page whose session expired goes to the login page too.
    return JSONResponse(
        {"detail": "Not logged in"}, status_code=401, headers={"HX-Redirect": api_paths.WEB_LOGIN}
    )
