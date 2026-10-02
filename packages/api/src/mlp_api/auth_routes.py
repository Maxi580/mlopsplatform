from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

from mlp_api.auth.session import log_in, set_session_cookie
from mlp_core import api_paths

router = APIRouter()


class Login(BaseModel):
    password: str


@router.post(api_paths.LOGIN)
def login(login: Login, request: Request, response: Response) -> dict:
    token = log_in(request, login.password)
    set_session_cookie(response, token)
    return {"token": token}


# Traefik's forwardAuth for the KFP UI, MLflow UI and Endpoints; require_login does the check.
@router.get(api_paths.VERIFY)
def verify() -> None:
    pass
