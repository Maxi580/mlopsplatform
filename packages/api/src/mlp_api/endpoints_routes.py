from fastapi import APIRouter, HTTPException, Request
from pydantic import Field

from mlp_api.endpoints.lifecycle import (
    EndpointNameTaken,
    list_endpoints,
    start_endpoint,
    stop_endpoint,
)
from mlp_core import api_paths, config
from mlp_core.endpoint_spec import ENDPOINT_NAME_PATTERN, EndpointSpec

router = APIRouter()


class EndpointStart(EndpointSpec):
    name: str = Field(
        pattern=f"^{ENDPOINT_NAME_PATTERN}$", max_length=config.ENDPOINT_NAME_MAX_LENGTH
    )


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
