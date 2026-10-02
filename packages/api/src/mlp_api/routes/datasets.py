import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi import Path as PathParameter
from starlette.concurrency import run_in_threadpool

from mlp_api.datasets import (
    dataset_key,
    delete_dataset_version,
    find_dataset_version,
    list_datasets,
    upload_dataset_version,
)
from mlp_core import api_paths
from mlp_core.pipeline_request.references import DATASET_NAME_PATTERN

router = APIRouter()

DatasetName = Annotated[str, PathParameter(pattern=f"^{DATASET_NAME_PATTERN}$")]


@router.get(api_paths.DATASETS)
def datasets(request: Request) -> list[dict]:
    return list_datasets(request.app.state.engine)


# The JSONL file is the raw request body, streamed to disk so its size doesn't matter.
@router.post(api_paths.DATASET_VERSIONS, status_code=201)
async def upload(name: DatasetName, request: Request) -> dict:
    state = request.app.state
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "data.jsonl"
        with path.open("wb") as file:
            async for chunk in request.stream():
                file.write(chunk)
        try:
            return await run_in_threadpool(
                upload_dataset_version, state.engine, state.object_store, name, path
            )
        except ValueError as error:
            raise HTTPException(422, f"Rejected {name}: {error}") from None


@router.get(api_paths.DATASET_DOWNLOAD)
def download(name: DatasetName, version: int, request: Request) -> dict:
    if find_dataset_version(request.app.state.engine, name, version) is None:
        raise HTTPException(404, f"Dataset {name} has no version {version}")
    return {"url": request.app.state.object_store.download_url(dataset_key(name, version))}


@router.delete(api_paths.DATASET_VERSION, status_code=204)
def delete(name: DatasetName, version: int, request: Request) -> Response:
    state = request.app.state
    if not delete_dataset_version(state.engine, state.object_store, name, version):
        raise HTTPException(404, f"Dataset {name} has no version {version}")
    return Response(status_code=204)
