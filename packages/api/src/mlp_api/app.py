import os
import threading
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from sqlalchemy import create_engine

from mlp_api import (
    auth_routes,
    base_models_routes,
    checkpoints_routes,
    datasets_routes,
    endpoints_routes,
    health_routes,
    model_cache_routes,
    models_routes,
    pipelines_routes,
    settings_routes,
    smoke_tests_routes,
    storage_routes,
)
from mlp_api.auth.session import require_login
from mlp_api.model_cache.janitor import evict_forever
from mlp_api.models.mlflow import MLflow
from mlp_api.pipelines.cluster import Cluster
from mlp_api.pipelines.hugging_face import HuggingFace
from mlp_api.pipelines.lifecycle import fail_unsubmitted_pipelines
from mlp_api.pipelines.reconciler import reconcile_forever
from mlp_api.storage.database import create_tables
from mlp_api.storage.object_store import ObjectStore
from mlp_core import config
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
    # Where steps put Base Models (HF_HOME) and benchmarks, mounted at the same path.
    app.state.model_cache = Path(config.MODEL_CACHE_PATH)
    app.state.engine = create_engine(os.environ["DATABASE_URL"])
    create_tables(app.state.engine)
    fail_unsubmitted_pipelines(app.state.engine)
    stop = threading.Event()
    for loop in (reconcile_forever, evict_forever):
        threading.Thread(target=loop, args=(app.state, stop), daemon=True).start()
    yield
    stop.set()
    app.state.engine.dispose()


app = FastAPI(title="MLOps Platform", lifespan=lifespan)
app.middleware("http")(require_login)
app.include_router(auth_routes.router)
app.include_router(base_models_routes.router)
app.include_router(checkpoints_routes.router)
app.include_router(datasets_routes.router)
app.include_router(endpoints_routes.router)
app.include_router(health_routes.router)
app.include_router(model_cache_routes.router)
app.include_router(models_routes.router)
app.include_router(pipelines_routes.router)
app.include_router(settings_routes.router)
app.include_router(smoke_tests_routes.router)
app.include_router(storage_routes.router)
