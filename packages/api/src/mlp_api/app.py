import os
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import create_engine

from mlp_api.database import create_tables
from mlp_api.hugging_face import HuggingFace
from mlp_api.object_store import ObjectStore
from mlp_api.routes import auth, datasets, health, pipelines
from mlp_api.session import require_login
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
app.include_router(auth.router)
app.include_router(datasets.router)
app.include_router(health.router)
app.include_router(pipelines.router)
