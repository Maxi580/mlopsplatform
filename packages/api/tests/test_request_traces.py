from datetime import UTC, datetime, timedelta

import pytest

from mlp_api.endpoints.environment import EndpointEnvironment
from mlp_core import api_paths
from mlp_core.endpoint_spec import ServingOptions, VllmModel
from mlp_core.endpoint_spec import vllm_args as shared_vllm_args

from .conftest import record_request_trace
from .test_endpoint_stats import qwen_on_the_hub  # noqa: F401
from .test_endpoints import pod, reconcile, restart, start, stop, vllm_args

EXPERIMENT = "endpoint/chat"


def vllm_env(cluster, name="chat") -> dict[str, str]:
    [vllm] = pod(cluster, name)["containers"]
    return {variable["name"]: variable.get("value") for variable in vllm["env"]}


def test_an_endpoints_vllm_sends_a_trace_per_request_to_its_experiment(
    logged_in_api,
    qwen_on_the_hub,  # noqa: F811
    cluster,
    model_registry,
):
    start(logged_in_api)

    assert "--otlp-traces-endpoint=http://mlflow.test/v1/traces" in vllm_args(cluster)
    assert "--collect-detailed-traces" not in " ".join(vllm_args(cluster))
    env = vllm_env(cluster)
    assert env["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] == "http/protobuf"
    assert env["OTEL_EXPORTER_OTLP_TRACES_HEADERS"] == f"x-mlflow-experiment-id={EXPERIMENT}"
    assert env["OTEL_SERVICE_NAME"] == "chat"
    assert EXPERIMENT in model_registry.experiments


def test_a_restarted_endpoint_still_sends_traces(
    logged_in_api,
    qwen_on_the_hub,  # noqa: F811
    cluster,
    model_registry,
):
    start(logged_in_api)
    stop(logged_in_api)

    restart(logged_in_api)

    assert "--otlp-traces-endpoint=http://mlflow.test/v1/traces" in vllm_args(cluster)


def test_an_endpoint_starts_without_tracing_while_mlflow_is_unreachable(
    logged_in_api,
    qwen_on_the_hub,  # noqa: F811
    cluster,
    model_registry,
    monkeypatch,
):
    def unreachable(name, create=True):
        raise ConnectionError("mlflow.test timed out")

    monkeypatch.setattr(model_registry, "experiment_id", unreachable)

    response = start(logged_in_api)

    assert response.status_code == 201, response.text
    assert not any(arg.startswith("--otlp") for arg in vllm_args(cluster))
    assert not any(variable.startswith("OTEL_") for variable in vllm_env(cluster))


def test_stages_and_teachers_serve_without_tracing():
    args = shared_vllm_args(ServingOptions(), VllmModel("/models/weights"), "teacher", 1)

    assert not any(arg.startswith("--otlp") for arg in args)


def test_the_endpoint_environment_knows_mlflow(platform_database):
    assert EndpointEnvironment.from_environment().mlflow_url == "http://mlflow.test"


@pytest.fixture
def now():
    return datetime.now(UTC)


QUESTION = "What's the weather in Rome? " + "x" * 100
CHAT = {
    "messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": QUESTION},
    ],
    "tools": [{"type": "function", "function": {"name": "weather"}}],
    "temperature": 0.7,
}
CALL = {"id": "call-1", "type": "function", "function": {"name": "weather", "arguments": "{}"}}
MESSAGE = {"role": "assistant", "content": "Sunny.", "reasoning_content": "Look it up."}
ANSWER = {
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {**MESSAGE, "tool_calls": [CALL]}}],
}


def log_trace(model_registry, trace_id, arrived, e2e, tags=None, **sampling):
    record_request_trace(
        model_registry, EXPERIMENT, trace_id, arrived, e2e, CHAT, ANSWER, tags, **sampling
    )


def requests(api, name="chat", **filters):
    response = api.get(api_paths.ENDPOINT_REQUESTS.format(name=name), params=filters)
    assert response.status_code == 200, response.text
    return response.json()


def trace_ids(api, **filters) -> list[str]:
    return [row["trace_id"] for row in requests(api, **filters)["requests"]]


def request_trace(api, trace_id="tr-1", name="chat"):
    return api.get(api_paths.ENDPOINT_REQUEST.format(name=name, trace_id=trace_id))


@pytest.fixture
def chat(logged_in_api, qwen_on_the_hub, cluster):  # noqa: F811
    start(logged_in_api)


def test_a_request_row_holds_its_breakdown_tags_and_prompt_start(
    logged_in_api, chat, model_registry, now
):
    arrived = now - timedelta(minutes=1)
    tags = {"source": "chat", "rating": "up", "judge_score": "4"}
    log_trace(model_registry, "tr-1", arrived, 0.5, tags, **{"gen_ai.request.max_tokens": 256})

    [row] = requests(logged_in_api)["requests"]

    assert row == {
        "trace_id": "tr-1",
        "arrived_at": arrived.isoformat().replace("+00:00", "Z"),
        "latency": 0.5,
        "end_to_end": 0.5,
        "time_to_first_token": 0.25,
        "queue_time": 0.125,
        "prefill_time": 0.0625,
        "decode_time": 0.25,
        "prompt_tokens": 12,
        "completion_tokens": 40,
        "temperature": None,
        "top_p": None,
        "max_tokens": 256,
        "source": "chat",
        "status": "ok",
        "rating": "up",
        "judge_score": 4.0,
        "prompt_preview": QUESTION[:80],
        "mlflow_url": f"/mlflow/#/experiments/{EXPERIMENT}/traces/tr-1",
    }


def test_requests_are_newest_or_slowest_first_in_pages_of_50(
    logged_in_api, chat, model_registry, now
):
    for index in range(51):
        log_trace(model_registry, f"tr-{index}", now - timedelta(seconds=index), 1 + index % 3)

    first = requests(logged_in_api)
    second = requests(logged_in_api, page_token=first["next_page_token"])

    assert len(first["requests"]) == 50
    assert [row["trace_id"] for row in first["requests"][:2]] == ["tr-0", "tr-1"]
    assert [row["trace_id"] for row in second["requests"]] == ["tr-50"]
    assert second["next_page_token"] is None
    assert trace_ids(logged_in_api, order="latency")[0] == "tr-2"
    assert len(model_registry.trace_reads[0]) == 50


@pytest.mark.parametrize(
    ("filters", "clause"),
    [
        ({"source": "chat"}, ("tag.source", "=", "chat")),
        ({"status": "error"}, ("tag.status", "=", "error")),
        ({"rating": "down"}, ("tag.rating", "=", "down")),
        ({"rating": "none"}, ("tag.rating", "IS NULL", None)),
        ({"score_min": 3}, ("feedback.judge", ">=", 3)),
        ({"score_max": 2}, ("feedback.judge", "<=", 2)),
        ({"min_latency_ms": 1500}, ("trace.execution_time_ms", ">=", 1500)),
        ({"from": "2026-10-27T10:00:00Z"}, ("trace.timestamp_ms", ">=", 1793095200000)),
        ({"to": "2026-10-27T10:00:00Z"}, ("trace.timestamp_ms", "<=", 1793095200000)),
    ],
)
def test_each_filter_becomes_an_mlflow_search_clause(
    logged_in_api, chat, model_registry, filters, clause
):
    log_trace(model_registry, "tr-1", datetime.now(UTC), 1.0)

    requests(logged_in_api, **filters)

    [(_, clauses, order_by, size, _)] = model_registry.trace_searches
    assert clauses == [clause]
    assert (order_by, size) == ("timestamp_ms DESC", 50)


def test_the_filters_find_the_matching_requests(logged_in_api, chat, model_registry, now):
    log_trace(model_registry, "tr-api", now - timedelta(hours=2), 0.5)
    log_trace(model_registry, "tr-chat", now - timedelta(minutes=5), 2.0, {"source": "chat"})
    log_trace(model_registry, "tr-down", now - timedelta(minutes=1), 0.5, {"rating": "down"})
    judged = {"assessment_name": "judge", "feedback": {"value": 4}}
    model_registry.trace_info("tr-chat")["assessments"].append(judged)

    assert trace_ids(logged_in_api, source="chat") == ["tr-chat"]
    assert trace_ids(logged_in_api, rating="none") == ["tr-chat", "tr-api"]
    assert trace_ids(logged_in_api, score_min=4, score_max=5) == ["tr-chat"]
    assert trace_ids(logged_in_api, min_latency_ms=1000) == ["tr-chat"]
    since = (now - timedelta(hours=1)).isoformat()
    assert trace_ids(logged_in_api, **{"from": since}) == ["tr-down", "tr-chat"]


@pytest.mark.parametrize(
    "filters", [{"source": "cli"}, {"rating": "meh"}, {"score_min": 6}, {"order": "name"}]
)
def test_filters_take_only_known_values(logged_in_api, chat, filters):
    url = api_paths.ENDPOINT_REQUESTS.format(name="chat")

    assert logged_in_api.get(url, params=filters).status_code == 422


def test_requests_without_any_trace_are_empty(logged_in_api, chat, model_registry):
    assert requests(logged_in_api) == {"requests": [], "next_page_token": None}
    assert model_registry.trace_reads == []


def test_a_stopped_endpoint_still_lists_its_requests(
    logged_in_api, chat, cluster, model_registry, now
):
    log_trace(model_registry, "tr-1", now - timedelta(minutes=1), 1.0)
    stop(logged_in_api)
    reconcile(logged_in_api)

    assert trace_ids(logged_in_api) == ["tr-1"]


def test_requests_of_an_unknown_endpoint_are_not_found(logged_in_api, model_registry):
    assert logged_in_api.get(api_paths.ENDPOINT_REQUESTS.format(name="nope")).status_code == 404


def test_a_request_shows_its_full_content_feedback_and_judge_score(
    logged_in_api, chat, model_registry, now
):
    log_trace(model_registry, "tr-1", now, 0.5, {"rating": "down", "judge_score": "2"})
    judged = {"assessment_name": "judge", "feedback": {"value": 2}, "rationale": "Made up."}
    model_registry.trace_info("tr-1")["assessments"].append(judged)

    response = request_trace(logged_in_api)

    assert response.status_code == 200, response.text
    found = response.json()
    assert found["messages"] == CHAT["messages"]
    assert found["tools"] == CHAT["tools"]
    assert (found["reasoning"], found["answer"]) == ("Look it up.", "Sunny.")
    assert found["tool_calls"] == [CALL]
    assert found["error"] is None
    assert (found["rating"], found["judge_score"]) == ("down", 2.0)
    assert found["judge_reason"] == "Made up."
    assert found["mlflow_url"] == f"/mlflow/#/experiments/{EXPERIMENT}/traces/tr-1"


def test_a_completion_shows_its_prompt_as_the_users_message(logged_in_api, chat, model_registry):
    answer = {"choices": [{"index": 0, "text": " upon a time"}]}
    record_request_trace(
        model_registry, EXPERIMENT, "tr-1", datetime.now(UTC), 0.5, {"prompt": "Once"}, answer
    )

    found = request_trace(logged_in_api).json()

    assert found["messages"] == [{"role": "user", "content": "Once"}]
    assert (found["answer"], found["prompt_preview"]) == (" upon a time", "Once")


def test_another_endpoints_request_is_not_found(logged_in_api, chat, model_registry):
    start(logged_in_api, name="other")
    log_trace(model_registry, "tr-1", datetime.now(UTC), 0.5)

    assert request_trace(logged_in_api, name="other").status_code == 404
    assert request_trace(logged_in_api, trace_id="tr-nope").status_code == 404
