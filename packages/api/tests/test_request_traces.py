from datetime import UTC, datetime, timedelta

import pytest

from mlp_api.endpoints.environment import EndpointEnvironment
from mlp_core import api_paths, config
from mlp_core.endpoint_spec import ServingOptions, VllmModel
from mlp_core.endpoint_spec import vllm_args as shared_vllm_args

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


def log_trace(model_registry, trace_id, arrived, e2e, **sampling):
    """A Request Trace in chat's experiment, shaped as MLflow 3.16 answers `traces/batchGet`."""
    attributes = {
        "gen_ai.latency.e2e": e2e,
        "gen_ai.latency.time_to_first_token": 0.25,
        "gen_ai.latency.time_in_queue": 0.125,
        "gen_ai.latency.time_in_model_prefill": 0.0625,
        "gen_ai.latency.time_in_model_decode": e2e - 0.25,
        "gen_ai.latency.time_in_model_inference": e2e - 0.125,
        # int64 values come as JSON strings.
        "gen_ai.usage.prompt_tokens": "12",
        "gen_ai.usage.completion_tokens": 40,
        "gen_ai.request.id": f"chatcmpl-{trace_id}",
        **sampling,
    }

    def any_value(value):
        if isinstance(value, str) and not value.isdigit():
            return {"string_value": value}
        return {"int_value": value} if isinstance(value, int | str) else {"double_value": value}

    model_registry.traces.setdefault(EXPERIMENT, []).append(
        {
            "trace_info": {
                "trace_id": trace_id,
                "request_time": arrived.isoformat().replace("+00:00", "Z"),
                "execution_duration": f"{e2e}s",
                "state": "OK",
            },
            "spans": [
                {
                    "name": "llm_request",
                    "attributes": [
                        {"key": key, "value": any_value(value)} for key, value in attributes.items()
                    ],
                }
            ],
        }
    )


def requests(api, span="24h", name="chat"):
    return api.get(api_paths.ENDPOINT_REQUESTS.format(name=name), params={"span": span})


@pytest.fixture
def chat(logged_in_api, qwen_on_the_hub, cluster):  # noqa: F811
    start(logged_in_api)


def test_requests_lists_the_slowest_and_the_most_recent_with_their_breakdown(
    logged_in_api, chat, model_registry, now
):
    log_trace(model_registry, "tr-old-slow", now - timedelta(hours=2), 9.5)
    log_trace(
        model_registry,
        "tr-new-fast",
        now - timedelta(minutes=1),
        0.5,
        **{"gen_ai.request.temperature": 0.7, "gen_ai.request.max_tokens": 256},
    )
    log_trace(model_registry, "tr-too-old", now - timedelta(days=2), 20.0)

    response = requests(logged_in_api)

    assert response.status_code == 200, response.text
    answer = response.json()
    assert [row["trace_id"] for row in answer["slowest"]] == ["tr-old-slow", "tr-new-fast"]
    assert [row["trace_id"] for row in answer["recent"]] == ["tr-new-fast", "tr-old-slow"]
    assert answer["recent"][0] == {
        "trace_id": "tr-new-fast",
        "arrived_at": (now - timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
        "end_to_end": 0.5,
        "time_to_first_token": 0.25,
        "queue_time": 0.125,
        "prefill_time": 0.0625,
        "decode_time": 0.25,
        "prompt_tokens": 12,
        "completion_tokens": 40,
        "temperature": 0.7,
        "top_p": None,
        "max_tokens": 256,
        "mlflow_url": f"/mlflow/#/experiments/{EXPERIMENT}/traces/tr-new-fast",
    }


def test_requests_searches_twice_and_reads_the_spans_once(logged_in_api, chat, model_registry, now):
    log_trace(model_registry, "tr-1", now - timedelta(minutes=1), 1.0)

    requests(logged_in_api, span="1h")

    [(_, slowest_since, slowest_order, size), (_, recent_since, recent_order, _)] = (
        model_registry.trace_searches
    )
    assert (slowest_order, recent_order) == ("execution_time_ms DESC", "timestamp_ms DESC")
    assert slowest_since == recent_since
    hour_ago = (now - timedelta(hours=1)).timestamp() * 1000
    assert hour_ago <= slowest_since <= hour_ago + 60_000
    assert size == config.ENDPOINT_REQUEST_LIST_SIZE
    assert model_registry.trace_reads == [["tr-1"]]


def test_requests_without_any_trace_are_empty(logged_in_api, chat, model_registry):
    response = requests(logged_in_api)

    assert response.json() == {"slowest": [], "recent": []}
    assert model_registry.trace_reads == []


def test_a_stopped_endpoint_still_lists_its_requests(
    logged_in_api, chat, cluster, model_registry, now
):
    log_trace(model_registry, "tr-1", now - timedelta(minutes=1), 1.0)
    stop(logged_in_api)
    reconcile(logged_in_api)

    response = requests(logged_in_api)

    assert [row["trace_id"] for row in response.json()["recent"]] == ["tr-1"]


def test_requests_of_an_unknown_endpoint_are_not_found(logged_in_api, model_registry):
    assert requests(logged_in_api, name="nope").status_code == 404


def test_requests_take_only_known_spans(logged_in_api, chat, model_registry):
    assert requests(logged_in_api, span="2y").status_code == 422
