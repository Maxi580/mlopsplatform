from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial

from fastapi import FastAPI, Request

from mlp_core import config
from mlp_core.settings import Settings
from mlp_sandbox.snippets import Snippet, SnippetResult, run_snippet


# One pool for all batches, so the replica never runs more snippets than it has CPUs and memory for.
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = Settings()
    with ThreadPoolExecutor(app.state.settings.sandbox_cpus_per_replica) as pool:
        app.state.pool = pool
        yield


app = FastAPI(title="Sandbox", lifespan=lifespan)


@app.post(config.SANDBOX_PATH)
def run_snippets(snippets: list[Snippet], request: Request) -> list[SnippetResult]:
    """Each snippet's result, in order."""
    state = request.app.state
    return list(state.pool.map(partial(run_snippet, settings=state.settings), snippets))
