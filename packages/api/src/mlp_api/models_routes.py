from typing import Annotated

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi import Path as PathParameter

from mlp_api.models.registry import delete_model_version, list_models
from mlp_core import api_paths
from mlp_core.pipeline_request.references import MODEL_NAME_PATTERN

router = APIRouter()

ModelName = Annotated[str, PathParameter(pattern=f"^{MODEL_NAME_PATTERN}$")]


@router.get(api_paths.MODELS)
def models(request: Request) -> list[dict]:
    return list_models(request.app.state.model_registry, request.app.state.object_store)


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
