import os
from functools import cache

import httpx2 as httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from mlp_core import config
from mlp_proxy.request_trace import RequestTrace

app = FastAPI(title="Endpoint Proxy")
# No timeout, as a long answer streams for as long as vLLM generates it.
client = httpx.AsyncClient(timeout=None)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"])
async def proxy(path: str, request: Request) -> Response:
    """vLLM's answer; a chat or completion request's is recorded in its Request Trace."""
    # 1. Where it goes and which experiment traces it, as the Endpoint Key check said.
    upstream = request.headers.get(config.PROXY_UPSTREAM_HEADER)
    experiment_id = request.headers.get(config.PROXY_EXPERIMENT_HEADER)
    if not upstream or not experiment_id:
        detail = "Only Endpoint routes reach the Endpoint Proxy, through the Endpoint Key check"
        return JSONResponse({"detail": detail}, status_code=502)
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in config.PROXY_DROPPED_REQUEST_HEADERS
    }
    upstream_request = client.build_request(
        request.method,
        f"{upstream}/{path}",
        params=request.query_params,
        headers=headers,
        content=await request.body(),
    )

    # 2. Anything but a chat or completion request passes through untraced.
    if request.method != "POST" or f"/{path}" not in config.PROXY_TRACED_PATHS:
        return streamed(await client.send(upstream_request, stream=True))

    # 3. The rest under the trace's root span, which vLLM's own span joins through `traceparent`.
    span = tracer(experiment_id).start_span(config.REQUEST_TRACE_ROOT_SPAN)
    TraceContextTextMapPropagator().inject(
        upstream_request.headers, trace.set_span_in_context(span)
    )
    source = request.headers.get(config.PROXY_SOURCE_HEADER)
    record = RequestTrace(span, upstream_request.content, source)
    trace_id = {config.TRACE_ID_HEADER: record.trace_id}
    try:
        answer = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as error:
        record.finish(502, f"vLLM did not answer: {error}")
        return JSONResponse({"detail": "The Endpoint did not answer"}, 502, headers=trace_id)
    return streamed(answer, record, trace_id)


def streamed(
    answer: httpx.Response, record: RequestTrace | None = None, headers: dict | None = None
) -> StreamingResponse:
    """vLLM's answer, streamed to the client as it arrives, and recorded once it ended."""

    async def chunks():
        try:
            async for chunk in answer.aiter_raw():
                if record:
                    record.add(chunk)
                yield chunk
        finally:
            await answer.aclose()
            if record:
                record.finish(answer.status_code)

    kept = {
        key: value
        for key, value in answer.headers.items()
        if key.lower() not in config.PROXY_DROPPED_RESPONSE_HEADERS
    }
    return StreamingResponse(chunks(), answer.status_code, headers={**kept, **(headers or {})})


# One exporter per experiment, as MLflow takes the experiment from the export's header; the span
# processor exports in the background, so tracing never delays or fails an answer.
@cache
def tracer(experiment_id: str) -> trace.Tracer:
    exporter = OTLPSpanExporter(
        endpoint=f"{os.environ['MLFLOW_URL']}{config.REQUEST_TRACES_PATH}",
        headers={config.REQUEST_TRACES_EXPERIMENT_HEADER: experiment_id},
    )
    provider = TracerProvider(resource=Resource({"service.name": "endpoint-proxy"}))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider.get_tracer(__name__)
