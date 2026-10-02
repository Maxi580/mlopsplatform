from fastapi import APIRouter, HTTPException, Request

from mlp_api.endpoints.lifecycle import (
    EndpointNameTaken,
    list_endpoints,
    start_endpoint,
    stop_endpoint,
)
from mlp_api.endpoints.pipeline_output import serve_pipeline_output
from mlp_core import api_paths
from mlp_core.endpoint_spec import EndpointName, EndpointSpec

router = APIRouter()


class EndpointStart(EndpointSpec):
    name: EndpointName


@router.get(api_paths.ENDPOINTS)
def endpoints(request: Request) -> list[dict]:
    return list_endpoints(request.app.state.engine)


@router.post(api_paths.ENDPOINTS, status_code=201)
def start(start: EndpointStart, request: Request) -> dict:
    spec = EndpointSpec.model_validate(start.model_dump(exclude={"name"}))
    try:
        return start_endpoint(request.app.state, start.name, spec)
    except EndpointNameTaken as error:
        raise HTTPException(409, str(error)) from None
    except ValueError as error:
        raise HTTPException(422, f"Can't serve {start.model}: {error}") from None
    except RuntimeError as error:
        raise HTTPException(502, str(error)) from None


@router.post(api_paths.STOP_ENDPOINT)
def stop(name: str, request: Request) -> dict:
    try:
        stop_endpoint(request.app.state.engine, request.app.state.cluster, name)
    except LookupError as error:
        raise HTTPException(404, str(error)) from None
    return {"name": name, "status": "stopped"}


# The Pipeline's `serve` step; require_login lets only that Pipeline's serve token through.
@router.post(api_paths.SERVE_PIPELINE, status_code=201)
def serve(id: int, request: Request) -> dict:
    try:
        return serve_pipeline_output(request.app.state, id)
    except LookupError as error:
        raise HTTPException(404, str(error)) from None
    except EndpointNameTaken as error:
        raise HTTPException(409, str(error)) from None
    except ValueError as error:
        raise HTTPException(422, str(error)) from None
    except RuntimeError as error:
        raise HTTPException(502, str(error)) from None
