from datetime import datetime

from mlp_api.models.mlflow import MLflow
from mlp_core import config


def endpoint_request_traces(model_registry: MLflow, name: str, span: str, now: datetime) -> dict:
    """The span's slowest and most recent Request Traces of the Endpoint name, newest first."""
    # 1. Both lists' trace IDs, without spans; none before the first trace makes the experiment.
    experiment = config.ENDPOINT_HISTORY_EXPERIMENT.format(name=name)
    experiment_id = model_registry.experiment_id(experiment, create=False)
    if experiment_id is None:
        return {"slowest": [], "recent": []}
    since_ms = round((now - config.ENDPOINT_HISTORY_SPANS[span]).timestamp() * 1000)
    size = config.ENDPOINT_REQUEST_LIST_SIZE
    trace_ids_by_list = {
        key: [
            info["trace_id"]
            for info in model_registry.search_traces(experiment_id, since_ms, order_by, size)
        ]
        for key, order_by in config.ENDPOINT_REQUEST_LISTS.items()
    }

    # 2. Their spans, read at once.
    trace_ids = list(dict.fromkeys(trace_ids_by_list["slowest"] + trace_ids_by_list["recent"]))
    traces = model_registry.batch_get_traces(trace_ids) if trace_ids else []
    rows = {
        row["trace_id"]: row
        for row in (request_row(trace, experiment_id) for trace in traces)
        if row is not None
    }
    return {
        key: [rows[trace_id] for trace_id in ids if trace_id in rows]
        for key, ids in trace_ids_by_list.items()
    }


def request_row(trace: dict, experiment_id: str) -> dict | None:
    """The request's arrival, breakdown and MLflow link; None if vLLM's span is missing."""
    attributes = span_attributes(trace, config.REQUEST_TRACE_SPAN)
    if attributes is None:
        return None
    info = trace["trace_info"]
    return {
        "trace_id": info["trace_id"],
        "arrived_at": info["request_time"],
        **{key: attributes.get(name) for key, name in config.REQUEST_TRACE_ATTRIBUTES.items()},
        "mlflow_url": config.MLFLOW_TRACE_URL.format(
            experiment_id=experiment_id, trace_id=info["trace_id"]
        ),
    }


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
