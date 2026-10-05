from pathlib import Path

import pytest

from mlp_api.endpoints.stats import quantile
from mlp_api.pipelines.hugging_face import HubModel
from mlp_core import api_paths

from .test_endpoints import COMMIT, QWEN, endpoints, reconcile, start

# Captured from the pinned vLLM image serving Qwen/Qwen2.5-0.5B-Instruct: right after it started,
# after a few chat requests, and after a few with ngram speculative decoding on.
CAPTURES = Path(__file__).parent / "vllm_metrics"
IDLE = (CAPTURES / "idle.txt").read_text()
SERVED = (CAPTURES / "served.txt").read_text()
SPECULATIVE = (CAPTURES / "speculative.txt").read_text()


@pytest.fixture
def qwen_on_the_hub(logged_in_api, hugging_face):
    hugging_face.models[QWEN] = HubModel(commit=COMMIT, needs_remote_code=False, model_type="qwen3")


@pytest.mark.parametrize(
    ("q", "expected"),
    [
        # 5 of 10 observations: 3 into the (0.1, 0.5] bucket, which holds 4.
        (0.5, 0.1 + 0.4 * 3 / 4),
        # Into the first bucket, which starts at 0.
        (0.1, 0.1 * 1 / 2),
        # Past the last finite bound.
        (0.95, 1.0),
    ],
)
def test_a_quantile_is_interpolated_inside_its_bucket(q, expected):
    cumulative = [(0.1, 2), (0.5, 6), (1.0, 8), (float("inf"), 10)]

    assert quantile(q, cumulative) == pytest.approx(expected)


def test_a_histogram_without_observations_has_no_quantile():
    assert quantile(0.5, [(0.1, 0), (float("inf"), 0)]) is None


def running(api, cluster, name="chat", metrics=SERVED):
    start(api, name=name)
    cluster.endpoint_states[name] = "running"
    if metrics:
        cluster.metrics[name] = metrics
    reconcile(api)


def test_a_running_endpoint_is_listed_with_its_load_generated_tokens_and_ttft(
    logged_in_api, qwen_on_the_hub, cluster
):
    running(logged_in_api, cluster)

    [listed] = endpoints(logged_in_api)

    stats = listed["stats"]
    assert {key: stats[key] for key in ("running", "waiting", "generation_tokens")} == {
        "running": 0,
        "waiting": 0,
        "generation_tokens": 63,
    }
    # 4 of 5 first tokens came within (0.01, 0.02] seconds.
    assert stats["time_to_first_token_p50"] == pytest.approx(0.01 + 0.01 * 2.5 / 4)
    assert stats["read_at"]


def test_a_pending_or_unreachable_endpoint_has_no_stats(logged_in_api, qwen_on_the_hub, cluster):
    start(logged_in_api, name="loading")
    cluster.metrics["loading"] = SERVED
    running(logged_in_api, cluster, name="stuck", metrics=None)
    running(logged_in_api, cluster, name="gone")
    logged_in_api.post(api_paths.STOP_ENDPOINT.format(name="gone"))

    listed = {e["name"]: e["stats"] for e in endpoints(logged_in_api)}

    assert listed == {"loading": None, "stuck": None, "gone": None}


def test_the_stats_route_explains_each_curated_value(logged_in_api, qwen_on_the_hub, cluster):
    running(logged_in_api, cluster)

    stats = logged_in_api.get(api_paths.ENDPOINT_STATS.format(name="chat")).json()

    sections = {s["title"]: {v["key"]: v for v in s["values"]} for s in stats["sections"]}
    assert list(sections) == ["Requests", "Tokens", "KV cache", "Latency"]
    assert sections["Requests"]["finished"]["value"] == 5
    assert sections["Tokens"]["prompt_tokens"]["value"] == 225
    assert sections["KV cache"]["prefix_cache_hit_rate"]["value"] == pytest.approx(128 / 225)
    kv_cache = sections["KV cache"]["kv_cache_usage"]
    assert "not the hit rate" in kv_cache["explanation"]
    ttft = sections["Latency"]["time_to_first_token"]
    assert ttft["label"] == "Time to first token"
    assert ttft["histogram"] == {
        "p50": pytest.approx(0.01625),
        "p95": pytest.approx(0.25 + 0.25 * 0.75),
        "mean": pytest.approx(0.30995750427246094 / 5),
        "sum": pytest.approx(0.30995750427246094),
        "count": 5,
    }


def test_the_stats_route_lists_every_metric_with_vllms_help_text(
    logged_in_api, qwen_on_the_hub, cluster
):
    running(logged_in_api, cluster)

    stats = logged_in_api.get(api_paths.ENDPOINT_STATS.format(name="chat")).json()

    metrics = {metric["name"]: metric for metric in stats["metrics"]}
    assert metrics["vllm:generation_tokens"] == {
        "name": "vllm:generation_tokens",
        "type": "counter",
        "help": "Number of generation tokens processed.",
        "value": 63,
    }
    assert metrics["vllm:request_prompt_tokens"]["histogram"]["count"] == 5
    # A summary has no buckets to estimate quantiles from.
    request_sizes = metrics["http_request_size_bytes"]["histogram"]
    assert (request_sizes["p50"], request_sizes["count"]) == (None, 5)
    # Only when another metric started.
    assert not [name for name in metrics if name.endswith("_created")]


def test_speculative_decoding_shows_once_it_is_on(logged_in_api, qwen_on_the_hub, cluster):
    running(logged_in_api, cluster, metrics=SPECULATIVE)

    stats = logged_in_api.get(api_paths.ENDPOINT_STATS.format(name="chat")).json()

    [speculative] = [s for s in stats["sections"] if s["title"] == "Speculative decoding"]
    values = {v["key"]: v["value"] for v in speculative["values"]}
    assert values == {"acceptance_rate": 20 / 24, "draft_tokens": 24, "accepted_tokens": 20}


@pytest.mark.parametrize(
    ("setup", "status"),
    [("unknown", 404), ("pending", 409), ("unreachable", 502)],
)
def test_the_stats_route_says_why_an_endpoint_has_none(
    logged_in_api, qwen_on_the_hub, cluster, setup, status
):
    if setup == "pending":
        start(logged_in_api)
    if setup == "unreachable":
        running(logged_in_api, cluster, metrics=None)

    response = logged_in_api.get(api_paths.ENDPOINT_STATS.format(name="chat"))

    assert response.status_code == status
    assert "chat" in response.json()["detail"]
