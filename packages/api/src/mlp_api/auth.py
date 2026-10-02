import time
from datetime import UTC, datetime

import jwt
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from mlp_api.account import password_matches
from mlp_api.config import (
    FAILED_LOGIN_WINDOW,
    JWT_ALGORITHM,
    MAX_FAILED_LOGINS,
    PUBLIC_PATHS,
    SESSION_COOKIE,
    TOKEN_LIFETIME,
)
from mlp_core import api_paths

router = APIRouter()


class Login(BaseModel):
    password: str


@router.post(api_paths.LOGIN)
def login(login: Login, request: Request, response: Response) -> dict:
    failed_logins = request.app.state.failed_logins[request.client.host]
    cutoff = time.monotonic() - FAILED_LOGIN_WINDOW.total_seconds()
    failed_logins[:] = [failed_at for failed_at in failed_logins if failed_at > cutoff]
    if len(failed_logins) >= MAX_FAILED_LOGINS:
        raise HTTPException(429, "Too many failed logins, try again later")
    if not password_matches(request.app.state.engine, login.password):
        failed_logins.append(time.monotonic())
        raise HTTPException(401, "Wrong password")

    expires = datetime.now(UTC) + TOKEN_LIFETIME
    token = jwt.encode({"exp": expires}, request.app.state.jwt_secret, algorithm=JWT_ALGORITHM)
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=int(TOKEN_LIFETIME.total_seconds()),
        httponly=True,
        secure=True,
        samesite="lax",
    )
    return {"token": token}


# Traefik's forwardAuth for the KFP UI, MLflow UI and Endpoints; require_login does the check.
@router.get(api_paths.VERIFY)
def verify() -> None:
    pass


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
    return JSONResponse({"detail": "Not logged in"}, status_code=401)
