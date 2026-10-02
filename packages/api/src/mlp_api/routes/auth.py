from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from mlp_api.account import password_matches
from mlp_api.config import SESSION_COOKIE, TOKEN_LIFETIME
from mlp_api.session import issue_token, record_failed_login, too_many_failed_logins
from mlp_core import api_paths

router = APIRouter()


class Login(BaseModel):
    password: str


@router.post(api_paths.LOGIN)
def login(login: Login, request: Request, response: Response) -> dict:
    failed_logins = request.app.state.failed_logins[request.client.host]
    if too_many_failed_logins(failed_logins):
        raise HTTPException(429, "Too many failed logins, try again later")
    if not password_matches(request.app.state.engine, login.password):
        record_failed_login(failed_logins)
        raise HTTPException(401, "Wrong password")

    token = issue_token(request.app.state.jwt_secret)
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
