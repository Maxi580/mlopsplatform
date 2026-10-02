import time
from datetime import UTC, datetime

import jwt
from fastapi import Request, Response
from fastapi.responses import JSONResponse, RedirectResponse

from mlp_core.config import (
    FAILED_LOGIN_WINDOW,
    JWT_ALGORITHM,
    MAX_FAILED_LOGINS,
    PUBLIC_PATHS,
    SESSION_COOKIE,
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
    # Browsers go to the Web UI's login, also via Traefik's forwardAuth for the KFP and MLflow UIs.
    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(WEB_UI_LOGIN_URL, status_code=303)
    return JSONResponse({"detail": "Not logged in"}, status_code=401)
