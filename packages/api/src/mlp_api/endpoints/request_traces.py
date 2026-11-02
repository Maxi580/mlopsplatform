from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from mlp_api.models.mlflow import MLflow
from mlp_core import config

Score = Field(None, ge=config.JUDGE_SCORES[0], le=config.JUDGE_SCORES[1])


class RequestFilters(BaseModel):
    """Which of an Endpoint's Request Traces the requests list shows, and in which order."""

    from_: datetime | None = Field(None, alias="from")
    to: datetime | None = None
    source: Literal[config.REQUEST_SOURCES] | None = None
    status: Literal["ok", "error"] | None = None
    rating: Literal[(*config.FEEDBACK_RATINGS, "none")] | None = None
    score_min: int | None = Score
    score_max: int | None = Score
    min_latency_ms: int | None = Field(None, ge=0)
    order: Literal[tuple(config.ENDPOINT_REQUEST_ORDERS)] = "time"
    page_token: str | None = None


def endpoint_requests(model_registry: MLflow, name: str, filters: RequestFilters) -> dict:
    """One page of the Endpoint name's Request Traces matching the filters, and the next page's
    token, or None after the last."""
    # 1. The page's trace IDs, without spans; none before the first trace makes the experiment.
    experiment = config.ENDPOINT_HISTORY_EXPERIMENT.format(name=name)
    experiment_id = model_registry.experiment_id(experiment, create=False)
    if experiment_id is None:
        return {"requests": [], "next_page_token": None}
    infos, next_page_token = model_registry.search_traces(
        experiment_id,
        request_clauses(filters),
        config.ENDPOINT_REQUEST_ORDERS[filters.order],
        config.ENDPOINT_REQUESTS_PAGE_SIZE,
        filters.page_token,
    )

    # 2. Their spans, read at once, in the page's order.
    trace_ids = [info["trace_id"] for info in infos]
    traces = model_registry.batch_get_traces(trace_ids) if trace_ids else []
    by_id = {trace["trace_info"]["trace_id"]: trace for trace in traces}
    rows = [request_row(by_id[found], experiment_id) for found in trace_ids if found in by_id]
    return {"requests": rows, "next_page_token": next_page_token}


def request_clauses(filters: RequestFilters) -> list[tuple[str, str, int | str | None]]:
    """The filters as MLflow trace search clauses; the score is the Judge's feedback assessment,
    as MLflow compares tags as text only."""
    clauses = []
    if filters.from_:
        clauses.append(("trace.timestamp_ms", ">=", unix_ms(filters.from_)))
    if filters.to:
        clauses.append(("trace.timestamp_ms", "<=", unix_ms(filters.to)))
    for tag in ("source", "status"):
        if getattr(filters, tag):
            clauses.append((f"tag.{tag}", "=", getattr(filters, tag)))
    if filters.rating == "none":
        clauses.append(("tag.rating", "IS NULL", None))
    elif filters.rating:
        clauses.append(("tag.rating", "=", filters.rating))
    if filters.score_min is not None:
        clauses.append(("feedback.judge", ">=", filters.score_min))
    if filters.score_max is not None:
        clauses.append(("feedback.judge", "<=", filters.score_max))
    if filters.min_latency_ms is not None:
        clauses.append(("trace.execution_time_ms", ">=", filters.min_latency_ms))
    return clauses


def unix_ms(at: datetime) -> int:
    return round(at.timestamp() * 1000)


def endpoint_request(model_registry: MLflow, name: str, trace_id: str) -> dict:
    """The Request Trace's row with its full content, Feedback and Judge Score; LookupError
    unless it is one of the Endpoint name's."""
    # 1. The trace, among the Endpoint's own only.
    experiment = config.ENDPOINT_HISTORY_EXPERIMENT.format(name=name)
    experiment_id = model_registry.experiment_id(experiment, create=False)
    traces = model_registry.batch_get_traces([trace_id]) if experiment_id else []
    if not traces or trace_experiment_id(traces[0]["trace_info"]) != experiment_id:
        raise LookupError(f"Endpoint {name} has no Request Trace {trace_id}")

    # 2. What the Endpoint Proxy recorded: the request, and the answer as vLLM gave it.
    root = span_attributes(traces[0], config.REQUEST_TRACE_ROOT_SPAN) or {}
    request = root.get(config.MLFLOW_SPAN_INPUTS) or {}
    response = root.get(config.MLFLOW_SPAN_OUTPUTS) or {}
    choice = (response.get("choices") or [{}])[0] if isinstance(response, dict) else {}
    message = choice.get("message", {})

    # 3. The Judge's reason beside its score.
    assessments = traces[0]["trace_info"].get("assessments", [])
    judged = [a for a in assessments if a["assessment_name"] == "judge"]
    return {
        **request_row(traces[0], experiment_id),
        "messages": request_messages(request),
        "tools": request.get("tools"),
        "reasoning": message.get("reasoning_content"),
        "answer": message.get("content", choice.get("text")),
        "tool_calls": message.get("tool_calls"),
        "error": root.get("error"),
        "judge_reason": judged[0].get("rationale") if judged else None,
    }


def request_row(trace: dict, experiment_id: str) -> dict:
    """The request's arrival, latency, vLLM's breakdown, tags, prompt start and MLflow link; the
    breakdown is empty while vLLM's span is missing, e.g. after an error before it."""
    info = trace["trace_info"]
    vllm = span_attributes(trace, config.REQUEST_TRACE_SPAN) or {}
    root = span_attributes(trace, config.REQUEST_TRACE_ROOT_SPAN) or {}
    tags = info.get("tags", {})
    score = tags.get("judge_score")
    return {
        "trace_id": info["trace_id"],
        "arrived_at": info["request_time"],
        # The Endpoint Proxy's span, from the request's arrival to the end of its answer.
        "latency": float(info["execution_duration"].removesuffix("s")),
        **{key: vllm.get(name) for key, name in config.REQUEST_TRACE_ATTRIBUTES.items()},
        "source": tags.get("source"),
        "status": tags.get("status"),
        "rating": tags.get("rating"),
        "judge_score": float(score) if score else None,
        "prompt_preview": prompt_preview(root.get(config.MLFLOW_SPAN_INPUTS) or {}),
        "mlflow_url": config.MLFLOW_TRACE_URL.format(
            experiment_id=experiment_id, trace_id=info["trace_id"]
        ),
    }


def trace_experiment_id(info: dict) -> str | None:
    return info.get("trace_location", {}).get("mlflow_experiment", {}).get("experiment_id")


def request_messages(request: dict) -> list[dict]:
    """A chat request's messages; a completion's prompt as one user message."""
    if not isinstance(request, dict):
        return []
    if "messages" in request:
        return request["messages"]
    prompt = request.get("prompt")
    return [{"role": "user", "content": prompt}] if prompt is not None else []


def prompt_preview(request: dict) -> str | None:
    """The start of the request's last user message."""
    users = [m for m in request_messages(request) if m.get("role") == "user"]
    if not users:
        return None
    content = users[-1].get("content")
    if isinstance(content, list):
        content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)[: config.PROMPT_PREVIEW_LENGTH]


def span_attributes(trace: dict, name: str) -> dict | None:
    """The attributes of the trace's span with the name, or None if it has none such."""
    span = next((s for s in trace.get("spans", []) if s["name"] == name), None)
    if span is None:
        return None
    return {attribute["key"]: any_value(attribute["value"]) for attribute in span["attributes"]}


# An OTel AnyValue as MLflow writes it to JSON; int64 values may come as strings, and the Endpoint
# Proxy's request and answer as key-value lists.
def any_value(value: dict):
    if "kvlist_value" in value:
        pairs = value["kvlist_value"].get("values", [])
        return {pair["key"]: any_value(pair.get("value", {})) for pair in pairs}
    if "array_value" in value:
        return [any_value(found) for found in value["array_value"].get("values", [])]
    if "int_value" in value:
        return int(value["int_value"])
    if "double_value" in value:
        return float(value["double_value"])
    return next(iter(value.values()), None)
