from datetime import UTC, datetime, timedelta

import pytest

from mlp_api.endpoints.history_retention import trim_endpoint_history

from .test_endpoint_stats import qwen_on_the_hub  # noqa: F401
from .test_endpoints import delete_endpoint, endpoints, start, stop

NOW = datetime(2026, 10, 12, 12, 0, tzinfo=UTC)
EXPERIMENT = "endpoint/chat"


def add_trace(model_registry, trace_id, arrived, experiment=EXPERIMENT):
    model_registry.traces.setdefault(experiment, []).append(
        {"trace_info": {"trace_id": trace_id, "request_time": arrived.isoformat()}}
    )


def trace_ids(model_registry, experiment=EXPERIMENT) -> list[str]:
    return [trace["trace_info"]["trace_id"] for trace in model_registry.traces.get(experiment, [])]


def add_day_runs(model_registry, *days, experiment=EXPERIMENT):
    model_registry.experiment_id(experiment)
    for day in days:
        model_registry.create_run(experiment, day, 0)


def test_history_older_than_30_days_is_deleted(logged_in_api, model_registry):
    add_day_runs(model_registry, "2026-09-11", "2026-09-12", "2026-10-12")
    add_day_runs(model_registry, "2026-09-01", experiment="endpoint/other")
    add_day_runs(model_registry, "2026-09-01", experiment="pipelines")
    add_trace(model_registry, "old", NOW - timedelta(days=30, minutes=1))
    add_trace(model_registry, "new", NOW - timedelta(days=29))

    trim_endpoint_history(logged_in_api.app.state, NOW)

    # The Run of 2026-09-12 still holds samples within the window.
    assert list(model_registry.experiments[EXPERIMENT]) == ["2026-09-12", "2026-10-12"]
    assert model_registry.experiments["endpoint/other"] == {}
    assert list(model_registry.experiments["pipelines"]) == ["2026-09-01"]
    assert trace_ids(model_registry) == ["new"]


def test_history_is_trimmed_at_most_once_an_hour(logged_in_api, model_registry):
    add_day_runs(model_registry, "2026-10-12")
    trim_endpoint_history(logged_in_api.app.state, NOW)
    add_trace(model_registry, "old", NOW - timedelta(days=31))

    trim_endpoint_history(logged_in_api.app.state, NOW + timedelta(minutes=59))
    kept = trace_ids(model_registry)
    trim_endpoint_history(logged_in_api.app.state, NOW + timedelta(hours=1))

    assert (kept, trace_ids(model_registry)) == (["old"], [])


def test_a_failed_trim_is_retried_the_next_hour(logged_in_api, model_registry, monkeypatch):
    add_day_runs(model_registry, "2026-09-01")

    def unreachable(experiment_id, max_timestamp_ms):
        raise ConnectionError("mlflow.test timed out")

    with monkeypatch.context() as patched:
        patched.setattr(model_registry, "delete_traces", unreachable)
        trim_endpoint_history(logged_in_api.app.state, NOW)
    trim_endpoint_history(logged_in_api.app.state, NOW + timedelta(hours=1))

    assert model_registry.experiments[EXPERIMENT] == {}


@pytest.fixture
def stopped_chat(logged_in_api, qwen_on_the_hub, cluster, model_registry):  # noqa: F811
    start(logged_in_api)
    stop(logged_in_api)
    add_day_runs(model_registry, "2026-10-12")
    add_trace(model_registry, "served", NOW)


def test_deleting_an_endpoint_deletes_its_traces_then_renames_and_deletes_its_experiment(
    logged_in_api, stopped_chat, model_registry
):
    response = delete_endpoint(logged_in_api)

    assert response.status_code == 204, response.text
    traces, rename, deleted = model_registry.deletions
    assert traces[:2] == ("traces", EXPERIMENT)
    assert traces[2] >= NOW.timestamp() * 1000
    assert rename[:2] == ("rename", EXPERIMENT)
    assert rename[2].startswith(f"{EXPERIMENT}/deleted-")
    assert deleted == ("experiment", EXPERIMENT)
    assert trace_ids(model_registry) == []


def test_a_reused_name_gets_a_fresh_experiment(logged_in_api, stopped_chat, model_registry):
    delete_endpoint(logged_in_api)

    start(logged_in_api)

    assert model_registry.experiments[EXPERIMENT] == {}


def test_deleting_an_endpoint_forgets_its_cached_reading_and_day_run(logged_in_api, stopped_chat):
    state = logged_in_api.app.state
    state.server_readings["chat"] = "reading"
    state.endpoint_day_runs["chat"] = ("2026-10-12", "run")

    delete_endpoint(logged_in_api)

    assert ("chat" in state.server_readings, "chat" in state.endpoint_day_runs) == (False, False)


def test_an_endpoint_is_deleted_while_mlflow_is_unreachable(
    logged_in_api, stopped_chat, model_registry, monkeypatch
):
    def unreachable(experiment_id, max_timestamp_ms):
        raise ConnectionError("mlflow.test timed out")

    monkeypatch.setattr(model_registry, "delete_traces", unreachable)

    response = delete_endpoint(logged_in_api)

    assert response.status_code == 204, response.text
    assert endpoints(logged_in_api) == []


def test_deleting_an_endpoint_without_history_touches_no_experiment(
    logged_in_api,
    qwen_on_the_hub,  # noqa: F811
    model_registry,
):
    start(logged_in_api)
    stop(logged_in_api)
    model_registry.experiments.clear()

    assert delete_endpoint(logged_in_api).status_code == 204
    assert model_registry.deletions == []
