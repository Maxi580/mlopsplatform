from fastapi import APIRouter, HTTPException, Request, Response

from mlp_api.endpoints.lifecycle import (
    EndpointNameTaken,
    delete_endpoint,
    find_endpoint,
    list_endpoints,
    start_endpoint,
    stop_endpoint,
)
from mlp_api.endpoints.stats import endpoint_stats_summary, fetch_endpoint_stats
from mlp_core import api_paths
from mlp_core.endpoint_spec import EndpointName, EndpointSpec

router = APIRouter()


class EndpointStart(EndpointSpec):
    name: EndpointName


@router.get(api_paths.ENDPOINTS)
def endpoints(request: Request) -> list[dict]:
    state = request.app.state
    listed = list_endpoints(state.engine, state.cluster.endpoint_environment.domain)
    running = [found["name"] for found in listed if found["status"] == "running"]
    stats = fetch_endpoint_stats(state.cluster, running)
    # A stopped Endpoint keeps its row, and its name may be running again.
    for found in listed:
        read = stats.get(found["name"]) if found["status"] == "running" else None
        found["stats"] = endpoint_stats_summary(read)
    return listed


@router.get(api_paths.ENDPOINT_STATS)
def stats(name: str, request: Request) -> dict:
    found = find_endpoint(request.app.state.engine, name)
    if found is None:
        raise HTTPException(404, f"No Endpoint {name} is running")
    if found.status != "running":
        detail = f"Endpoint {name} is {found.status}; it has stats once vLLM is ready"
        raise HTTPException(409, detail)
    read = fetch_endpoint_stats(request.app.state.cluster, [name])[name]
    if read is None:
        raise HTTPException(502, f"Endpoint {name} did not answer with its metrics")
    return read


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


@router.delete(api_paths.ENDPOINT, status_code=204)
def delete(name: str, request: Request) -> Response:
    try:
        delete_endpoint(request.app.state.engine, name)
    except LookupError as error:
        raise HTTPException(404, str(error)) from None
    except ValueError as error:
        raise HTTPException(409, str(error)) from None
    return Response(status_code=204)
