from typing import Annotated

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi import Path as PathParameter
from pydantic import BaseModel, ConfigDict, Field

from mlp_api.models.registry import delete_model_version, list_models, model_version_files
from mlp_api.models.uploads import complete_model_upload, start_model_upload
from mlp_core import api_paths
from mlp_core.pipeline_request.references import (
    MODEL_NAME_PATTERN,
    BaseModelReference,
    ModelReference,
)

router = APIRouter()

ModelName = Annotated[str, PathParameter(pattern=f"^{MODEL_NAME_PATTERN}$")]


class UploadedFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    size_bytes: int = Field(ge=0)


class ModelUpload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=f"^{MODEL_NAME_PATTERN}$", max_length=63)
    files: list[UploadedFile] = Field(min_length=1)
    # An Adapter's base: a Base Model or a full-weight Model Version.
    base: BaseModelReference | ModelReference | None = None
    # vLLM's tool-call parser, for a `model_type` the platform has none for.
    tool_parser: str | None = Field(None, pattern=r"^\w+$")


@router.get(api_paths.MODELS)
def models(request: Request) -> list[dict]:
    return list_models(request.app.state.model_registry, request.app.state.object_store)


# The files go straight to the object store, in parts; only their names pass through here.
@router.post(api_paths.MODEL_UPLOADS, status_code=201)
def start_upload(upload: ModelUpload, request: Request) -> dict:
    state = request.app.state
    files = {file.path: file.size_bytes for file in upload.files}
    if len(files) < len(upload.files):
        raise HTTPException(422, "Every file path must be unique")
    try:
        return start_model_upload(
            state.engine,
            state.object_store,
            state.model_registry,
            state.hugging_face,
            upload.name,
            files,
            upload.base,
            upload.tool_parser,
        )
    except ValueError as error:
        raise HTTPException(422, f"Rejected {upload.name}: {error}") from None


@router.post(api_paths.MODEL_UPLOAD_COMPLETE, status_code=201)
def complete_upload(id: str, request: Request) -> dict:
    state = request.app.state
    try:
        return complete_model_upload(state.engine, state.object_store, state.model_registry, id)
    except LookupError as error:
        raise HTTPException(404, str(error)) from None
    except ValueError as error:
        raise HTTPException(422, f"Rejected the upload: {error}") from None


@router.get(api_paths.MODEL_VERSION_FILES)
def files(name: ModelName, version: int, request: Request) -> dict:
    state = request.app.state
    try:
        return {
            "files": model_version_files(state.model_registry, state.object_store, name, version)
        }
    except LookupError as error:
        raise HTTPException(404, str(error)) from None


@router.delete(api_paths.MODEL_VERSION, status_code=204)
def delete(name: ModelName, version: int, request: Request) -> Response:
    state = request.app.state
    try:
        delete_model_version(state.engine, state.model_registry, state.object_store, name, version)
    except LookupError as error:
        raise HTTPException(404, str(error)) from None
    except ValueError as error:
        raise HTTPException(409, str(error)) from None
    return Response(status_code=204)
