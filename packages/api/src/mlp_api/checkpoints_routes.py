from fastapi import APIRouter, HTTPException, Request, Response

from mlp_api.checkpoints.registry import delete_checkpoint, list_checkpoints
from mlp_core import api_paths

router = APIRouter()


@router.get(api_paths.CHECKPOINTS)
def checkpoints(request: Request) -> list[dict]:
    return list_checkpoints(request.app.state.engine, request.app.state.object_store)


@router.delete(api_paths.CHECKPOINT, status_code=204)
def delete(pipeline_id: int, phase_index: int, request: Request) -> Response:
    state = request.app.state
    try:
        delete_checkpoint(state.engine, state.object_store, pipeline_id, phase_index)
    except LookupError as error:
        raise HTTPException(404, str(error)) from None
    except ValueError as error:
        raise HTTPException(409, str(error)) from None
    return Response(status_code=204)
