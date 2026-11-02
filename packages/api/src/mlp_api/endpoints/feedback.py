from datetime import datetime

from mlp_api.endpoints.lifecycle import traces_experiment_id
from mlp_api.endpoints.stats_history import day_run_id, round_ms
from mlp_core import config


def record_feedback(state, name: str, trace_id: str, rating: str, now: datetime) -> None:
    """Stores the rating on the Endpoint's Request Trace, replacing an earlier one, and counts it
    in its Stats History; LookupError unless the trace is one of the Endpoint's."""
    # 1. The trace, among the Endpoint's own only.
    registry = state.model_registry
    info = registry.trace_info(trace_id)
    if info is None or trace_experiment_id(info) != traces_experiment_id(state, name):
        raise LookupError(f"Endpoint {name} has no Request Trace {trace_id}")

    # 2. The vote as the trace's assessment, and as its tag the requests list filters on.
    previous = info.get("tags", {}).get("rating")
    registry.set_feedback(info, "rating", rating, "HUMAN")
    registry.set_trace_tag(trace_id, "rating", rating)

    # 3. Counted in the day Run; a changed vote takes the old one back, so the sums stay right.
    if previous == rating:
        return
    counts = {config.FEEDBACK_METRIC.format(rating=rating): 1}
    if previous:
        counts[config.FEEDBACK_METRIC.format(rating=previous)] = -1
    run_id = day_run_id(state, name, now)
    registry.log_metrics(run_id, counts, int(now.timestamp()), round_ms(now))


def trace_experiment_id(info: dict) -> str | None:
    return info.get("trace_location", {}).get("mlflow_experiment", {}).get("experiment_id")
