import os
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from sqlalchemy import create_engine, text

from mlp_api import auth
from mlp_api.database import upgrade_database
from mlp_core import api_paths
from mlp_core.settings import Settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = Settings()
    app.state.jwt_secret = os.environ["JWT_SECRET"]
    # In memory per client IP, which works because the API runs as one replica.
    app.state.failed_logins = defaultdict(list)
    app.state.engine = create_engine(os.environ["DATABASE_URL"])
    upgrade_database(app.state.engine)
    yield
    app.state.engine.dispose()


app = FastAPI(title="MLOps Platform", lifespan=lifespan)
app.middleware("http")(auth.require_login)
app.include_router(auth.router)


@app.get(api_paths.HEALTH)
def health(request: Request) -> dict:
    with request.app.state.engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return {"status": "ok"}
