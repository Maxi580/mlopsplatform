import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi import Path as PathParameter
from starlette.concurrency import run_in_threadpool

from mlp_api.datasets.pipeline_output import register_distillation_dataset
from mlp_api.datasets.registry import (
    delete_dataset_version,
    find_dataset_version,
    list_datasets,
    upload_dataset_version,
)
from mlp_core import api_paths
from mlp_core.pipeline_request.references import DATASET_NAME_PATTERN, dataset_key

router = APIRouter()

DatasetName = Annotated[str, PathParameter(pattern=f"^{DATASET_NAME_PATTERN}$")]


@router.get(api_paths.DATASETS)
def datasets(request: Request) -> list[dict]:
    return list_datasets(request.app.state.engine)


@router.post(api_paths.DATASET_VERSIONS, status_code=201)
async def upload(name: DatasetName, request: Request) -> dict:
    state = request.app.state
    with tempfile.TemporaryDirectory() as directory:
        path = await received_file(request, Path(directory))
        try:
            return await run_in_threadpool(
                upload_dataset_version, state.engine, state.object_store, name, path
            )
        except ValueError as error:
            raise HTTPException(422, f"Rejected {name}: {error}") from None


# The Pipeline's `distill` step; require_login lets only that Pipeline's step token through.
@router.post(api_paths.DISTILL_PIPELINE, status_code=201)
async def distill(id: int, request: Request) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        path = await received_file(request, Path(directory))
        try:
            reference = await run_in_threadpool(
                register_distillation_dataset, request.app.state, id, path
            )
        except LookupError as error:
            raise HTTPException(404, str(error)) from None
        except ValueError as error:
            raise HTTPException(422, f"Rejected the Distillation Dataset: {error}") from None
    return {"dataset": reference}


async def received_file(request: Request, directory: Path) -> Path:
    """The JSONL file of the request body, streamed to disk so its size doesn't matter."""
    path = directory / "data.jsonl"
    with path.open("wb") as file:
        async for chunk in request.stream():
            file.write(chunk)
    return path


@router.get(api_paths.DATASET_DOWNLOAD)
def download(name: DatasetName, version: int, request: Request) -> dict:
    if find_dataset_version(request.app.state.engine, name, version) is None:
        raise HTTPException(404, f"Dataset {name} has no version {version}")
    return {"url": request.app.state.object_store.download_url(dataset_key(name, version))}


@router.delete(api_paths.DATASET_VERSION, status_code=204)
def delete(name: DatasetName, version: int, request: Request) -> Response:
    state = request.app.state
    try:
        delete_dataset_version(state.engine, state.object_store, name, version)
    except LookupError as error:
        raise HTTPException(404, str(error)) from None
    except ValueError as error:
        raise HTTPException(409, str(error)) from None
    return Response(status_code=204)
