from datetime import UTC, datetime
from typing import Annotated, Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel

from mlp_api.endpoints.feedback import record_feedback
from mlp_api.endpoints.keys import endpoint_opened_by, refresh_endpoint_key
from mlp_api.endpoints.lifecycle import (
    EndpointConflict,
    delete_endpoint,
    endpoint_summary,
    find_endpoint,
    list_endpoints,
    restart_endpoint,
    start_endpoint,
    stop_endpoint,
    traces_experiment_id,
)
from mlp_api.endpoints.request_traces import RequestFilters, endpoint_request, endpoint_requests
from mlp_api.endpoints.stats import endpoint_stats_summary, fetch_endpoint_stats
from mlp_api.endpoints.stats_history import endpoint_history
from mlp_api.judge.judging import Judging, set_endpoint_judging
from mlp_api.pipelines.compiler import endpoint_service_url
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
    for found in listed:
        read = stats.get(found["name"]) if found["status"] == "running" else None
        found["stats"] = endpoint_stats_summary(read)
    return listed


@router.get(api_paths.ENDPOINT_STATS)
def stats(name: str, request: Request) -> dict:
    found = find_endpoint(request.app.state.engine, name)
    if found is None:
        raise HTTPException(404, f"No Endpoint {name}")
    if found.status != "running":
        detail = f"Endpoint {name} is {found.status}; it has stats once vLLM is ready"
        raise HTTPException(409, detail)
    read = fetch_endpoint_stats(request.app.state.cluster, [name])[name]
    if read is None:
        raise HTTPException(502, f"Endpoint {name} did not answer with its metrics")
    return read


# Stopped Endpoints have their history too.
@router.get(api_paths.ENDPOINT_HISTORY)
def history(name: str, request: Request, span: str = "live") -> dict:
    if span not in config.ENDPOINT_HISTORY_SPANS:
        raise HTTPException(422, f"span is one of {', '.join(config.ENDPOINT_HISTORY_SPANS)}")
    if find_endpoint(request.app.state.engine, name) is None:
        raise HTTPException(404, f"No Endpoint {name}")
    return endpoint_history(request.app.state.model_registry, name, span, datetime.now(UTC))


# Stopped Endpoints have their Request Traces too.
@router.get(api_paths.ENDPOINT_REQUESTS)
def requests(name: str, request: Request, filters: Annotated[RequestFilters, Query()]) -> dict:
    if find_endpoint(request.app.state.engine, name) is None:
        raise HTTPException(404, f"No Endpoint {name}")
    return endpoint_requests(request.app.state.model_registry, name, filters)


@router.get(api_paths.ENDPOINT_REQUEST)
def request_trace(name: str, trace_id: str, request: Request) -> dict:
    try:
        return endpoint_request(request.app.state.model_registry, name, trace_id)
    except LookupError as error:
        raise HTTPException(404, str(error)) from None


def refuse_the_judge(name: str) -> None:
    """409 for the Judge, which the API starts and stops itself as judging is switched."""
    if name == config.JUDGE_ENDPOINT_NAME:
        raise HTTPException(409, f"{name} is the Judge's name; switch judging in its settings")


@router.post(api_paths.ENDPOINTS, status_code=201)
def start(start: EndpointStart, request: Request) -> dict:
    refuse_the_judge(start.name)
    spec = EndpointSpec.model_validate(start.model_dump(exclude={"name"}))
    try:
        return start_endpoint(request.app.state, start.name, spec)
    except EndpointConflict as error:
        raise HTTPException(409, str(error)) from None
    except ValueError as error:
        raise HTTPException(422, f"Can't serve {start.model}: {error}") from None
    except RuntimeError as error:
        raise HTTPException(502, str(error)) from None


@router.post(api_paths.STOP_ENDPOINT)
def stop(name: str, request: Request) -> dict:
    refuse_the_judge(name)
    try:
        stop_endpoint(request.app.state.engine, request.app.state.cluster, name)
    except LookupError as error:
        raise HTTPException(404, str(error)) from None
    return {"name": name, "status": "stopped"}


@router.post(api_paths.RESTART_ENDPOINT)
def restart(name: str, request: Request) -> dict:
    refuse_the_judge(name)
    try:
        return restart_endpoint(request.app.state, name)
    except LookupError as error:
        raise HTTPException(404, str(error)) from None
    except EndpointConflict as error:
        raise HTTPException(409, str(error)) from None
    except ValueError as error:
        raise HTTPException(422, f"Can't serve Endpoint {name} again: {error}") from None
    except RuntimeError as error:
        raise HTTPException(502, str(error)) from None


@router.delete(api_paths.ENDPOINT, status_code=204)
def delete(name: str, request: Request) -> Response:
    refuse_the_judge(name)
    try:
        delete_endpoint(request.app.state, name)
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
def verify_endpoint_key(request: Request, response: Response) -> dict:
    """Where the Endpoint Proxy sends and traces the request, as headers; 401 without its key."""
    # 1. The Endpoint's UUID, the path segment after the route prefix.
    path = urlsplit(request.headers.get("x-forwarded-uri", "")).path
    prefix = config.ENDPOINT_PATH_PREFIX.split("{")[0]
    endpoint_uuid = ""
    if path.startswith(prefix):
        endpoint_uuid = path.removeprefix(prefix).split("/")[0]

    # 2. Whether the bearer token opens that Endpoint.
    state = request.app.state
    name = opened_endpoint(request, endpoint_uuid)

    # 3. Its vLLM and experiment, which Traefik puts over whatever the client sent in them.
    experiment_id = traces_experiment_id(state, name) or config.PROXY_UNTRACED
    namespace = state.cluster.endpoint_environment.namespace
    response.headers[config.PROXY_UPSTREAM_HEADER] = endpoint_service_url(name, namespace)
    response.headers[config.PROXY_EXPERIMENT_HEADER] = experiment_id
    return {}


@router.put(api_paths.ENDPOINT_JUDGING)
def judging(name: str, judging: Judging, request: Request) -> dict:
    refuse_the_judge(name)
    engine = request.app.state.engine
    try:
        set_endpoint_judging(engine, name, judging, datetime.now(UTC))
    except LookupError as error:
        raise HTTPException(404, str(error)) from None
    domain = request.app.state.cluster.endpoint_environment.domain
    return endpoint_summary(find_endpoint(engine, name), domain)


class Feedback(BaseModel):
    trace_id: str
    rating: Literal[config.FEEDBACK_RATINGS]


# Under the Endpoint's URL, so its users rate answers with what they call it with.
@router.post(api_paths.ENDPOINT_FEEDBACK)
def feedback(uuid: str, feedback: Feedback, request: Request) -> dict:
    state = request.app.state
    name = opened_endpoint(request, uuid)
    try:
        record_feedback(state, name, feedback.trace_id, feedback.rating, datetime.now(UTC))
    except LookupError as error:
        raise HTTPException(404, str(error)) from None
    return feedback.model_dump()


def opened_endpoint(request: Request, endpoint_uuid: str) -> str:
    """The name of the Endpoint whose key is the request's bearer token; else 401."""
    authorization = request.headers.get("authorization", "")
    key = authorization.removeprefix("Bearer ") if authorization.startswith("Bearer ") else ""
    name = endpoint_opened_by(request.app.state.engine, endpoint_uuid, key)
    if name is None:
        raise HTTPException(401, "This Endpoint needs its Endpoint Key as the bearer token")
    return name
