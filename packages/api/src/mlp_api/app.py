import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from sqlalchemy import create_engine, text

from mlp_api.database import upgrade_database
from mlp_core.settings import Settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = Settings()
    app.state.engine = create_engine(os.environ["DATABASE_URL"])
    upgrade_database(app.state.engine)
    yield
    app.state.engine.dispose()


app = FastAPI(title="MLOps Platform", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict:
    with request.app.state.engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return {"status": "ok"}
