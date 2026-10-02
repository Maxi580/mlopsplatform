from fastapi import APIRouter, Request
from sqlalchemy import text

from mlp_core import api_paths

router = APIRouter()


@router.get(api_paths.HEALTH)
def health(request: Request) -> dict:
    with request.app.state.engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return {"status": "ok"}
