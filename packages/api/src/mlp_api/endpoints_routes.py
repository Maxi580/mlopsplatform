from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request, Response

from mlp_api.endpoints.keys import endpoint_key_opens, refresh_endpoint_key
from mlp_api.endpoints.lifecycle import (
    EndpointNameTaken,
    delete_endpoint,
    endpoint_summary,
    find_endpoint,
    list_endpoints,
    start_endpoint,
    stop_endpoint,
)
from mlp_api.endpoints.stats import endpoint_stats_summary, fetch_endpoint_stats
from mlp_core import api_paths, config
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


@router.post(api_paths.REFRESH_ENDPOINT_KEY)
def refresh_key(name: str, request: Request) -> dict:
    engine = request.app.state.engine
    try:
        refresh_endpoint_key(engine, name)
    except LookupError as error:
        raise HTTPException(404, str(error)) from None
    domain = request.app.state.cluster.endpoint_environment.domain
    return endpoint_summary(find_endpoint(engine, name), domain)


# Traefik's forwardAuth on every Endpoint route, which it sends the request's path and headers.
@router.get(api_paths.VERIFY_ENDPOINT_KEY)
def verify_endpoint_key(request: Request) -> dict:
    path = urlsplit(request.headers.get("x-forwarded-uri", "")).path
    prefix = config.ENDPOINT_PATH_PREFIX.split("{")[0]
    endpoint_uuid = path.removeprefix(prefix).split("/")[0] if path.startswith(prefix) else ""
    authorization = request.headers.get("authorization", "")
    key = authorization.removeprefix("Bearer ") if authorization.startswith("Bearer ") else ""
    if not endpoint_key_opens(request.app.state.engine, endpoint_uuid, key):
        raise HTTPException(401, "This Endpoint needs its Endpoint Key as the bearer token")
    return {}
