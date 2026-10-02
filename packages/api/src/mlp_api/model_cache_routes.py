from fastapi import APIRouter, HTTPException, Request, Response

from mlp_api.model_cache.janitor import free_cached_base_model, list_cached_base_models
from mlp_core import api_paths
from mlp_core.pipeline_request.references import BaseModelReference
from mlp_core.settings import size_in_bytes

router = APIRouter()


@router.get(api_paths.CACHED_BASE_MODELS)
def cached_base_models(request: Request) -> dict:
    state = request.app.state
    return {
        "base_models": [
            {
                "reference": entry.reference,
                "size_bytes": entry.size_bytes,
                "last_used": entry.last_used.isoformat(),
            }
            for entry in list_cached_base_models(state.model_cache)
        ],
        "capacity_bytes": size_in_bytes(state.settings.model_cache_size),
    }


# A Reference holds slashes, so it travels as a query parameter rather than in the path.
@router.delete(api_paths.CACHED_BASE_MODELS, status_code=204)
def free(reference: BaseModelReference, request: Request) -> Response:
    state = request.app.state
    try:
        free_cached_base_model(state.engine, state.model_cache, reference)
    except LookupError as error:
        raise HTTPException(404, str(error)) from None
    except ValueError as error:
        raise HTTPException(409, str(error)) from None
    return Response(status_code=204)
