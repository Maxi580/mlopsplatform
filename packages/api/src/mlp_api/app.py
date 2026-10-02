import os
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import create_engine

from mlp_api import auth_routes, datasets_routes, health_routes, pipelines_routes
from mlp_api.auth.session import require_login
from mlp_api.database import create_tables
from mlp_api.datasets.object_store import ObjectStore
from mlp_api.pipelines.hugging_face import HuggingFace
from mlp_core.settings import Settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = Settings()
    app.state.hugging_face = HuggingFace()
    app.state.object_store = ObjectStore()
    app.state.jwt_secret = os.environ["JWT_SECRET"]
    # In memory per client IP, which works because the API runs as one replica.
    app.state.failed_logins = defaultdict(list)
    app.state.engine = create_engine(os.environ["DATABASE_URL"])
    create_tables(app.state.engine)
    yield
    app.state.engine.dispose()


app = FastAPI(title="MLOps Platform", lifespan=lifespan)
app.middleware("http")(require_login)
app.include_router(auth_routes.router)
app.include_router(datasets_routes.router)
app.include_router(health_routes.router)
app.include_router(pipelines_routes.router)
