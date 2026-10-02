import os
import threading
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import create_engine

from mlp_api import (
    auth_routes,
    datasets_routes,
    health_routes,
    models_routes,
    pipelines_routes,
    settings_routes,
    storage_routes,
)
from mlp_api.auth.session import require_login
from mlp_api.database import create_tables
from mlp_api.models.mlflow import MLflow
from mlp_api.object_store import ObjectStore
from mlp_api.pipelines.cluster import Cluster
from mlp_api.pipelines.hugging_face import HuggingFace
from mlp_api.pipelines.lifecycle import fail_unsubmitted_pipelines
from mlp_api.pipelines.reconciler import reconcile_forever
from mlp_core.settings import Settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = Settings()
    app.state.hugging_face = HuggingFace()
    app.state.object_store = ObjectStore()
    app.state.model_registry = MLflow()
    app.state.jwt_secret = os.environ["JWT_SECRET"]
    # In memory per client IP, which works because the API runs as one replica.
    app.state.failed_logins = defaultdict(list)
    app.state.cluster = Cluster()
    app.state.engine = create_engine(os.environ["DATABASE_URL"])
    create_tables(app.state.engine)
    fail_unsubmitted_pipelines(app.state.engine)
    stop = threading.Event()
    threading.Thread(target=reconcile_forever, args=(app.state, stop), daemon=True).start()
    yield
    stop.set()
    app.state.engine.dispose()


app = FastAPI(title="MLOps Platform", lifespan=lifespan)
app.middleware("http")(require_login)
app.include_router(auth_routes.router)
app.include_router(datasets_routes.router)
app.include_router(health_routes.router)
app.include_router(models_routes.router)
app.include_router(pipelines_routes.router)
app.include_router(settings_routes.router)
app.include_router(storage_routes.router)
