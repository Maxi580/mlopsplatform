from fastapi import APIRouter, HTTPException, Request, Response

from mlp_api.model_cache.janitor import free_cache_entry, list_cache_entries
from mlp_core import api_paths
from mlp_core.settings import size_in_bytes

router = APIRouter()


@router.get(api_paths.MODEL_CACHE)
def model_cache(request: Request) -> dict:
    state = request.app.state
    return {
        "entries": [
            {
                "kind": entry.kind,
                "reference": entry.reference,
                "size_bytes": entry.size_bytes,
                "last_used": entry.last_used.isoformat(),
            }
            for entry in list_cache_entries(state.model_cache)
        ],
        "capacity_bytes": size_in_bytes(state.settings.model_cache_size),
    }


# A Reference holds slashes, so it travels as a query parameter rather than in the path.
@router.delete(api_paths.MODEL_CACHE, status_code=204)
def free(reference: str, request: Request) -> Response:
    state = request.app.state
    try:
        free_cache_entry(state.engine, state.model_cache, reference)
    except LookupError as error:
        raise HTTPException(404, str(error)) from None
    except ValueError as error:
        raise HTTPException(409, str(error)) from None
    return Response(status_code=204)
