from datetime import UTC, datetime, timedelta

import pytest

from mlp_api.endpoints.stats_history import sample_endpoint_stats
from mlp_core import api_paths, config

from .test_endpoint_stats import IDLE, SERVED, qwen_on_the_hub, running  # noqa: F401
from .test_endpoints import reconcile, stop

T0 = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)


def metrics_text(running=0, waiting=0, finished=0, hits=0, queries=0, ttft=(0, 0, 0), ttft_sum=0):
    """A vLLM `/metrics` with a few curated values; `ttft` are the cumulative bucket counts."""
    buckets = "\n".join(
        f'vllm:time_to_first_token_seconds_bucket{{engine="0",le="{bound}"}} {count}'
        for bound, count in zip(("0.1", "0.5", "+Inf"), ttft, strict=True)
    )
    return f"""# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{{engine="0"}} {running}
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{{engine="0"}} {waiting}
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{{engine="0"}} {0.5 if running else 0}
# TYPE vllm:request_success_total counter
vllm:request_success_total{{engine="0",finished_reason="stop"}} {finished}
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{{engine="0"}} {hits}
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{{engine="0"}} {queries}
# TYPE vllm:time_to_first_token_seconds histogram
{buckets}
vllm:time_to_first_token_seconds_sum{{engine="0"}} {ttft_sum}
vllm:time_to_first_token_seconds_count{{engine="0"}} {ttft[-1]}
"""


@pytest.fixture
def chat(logged_in_api, qwen_on_the_hub, cluster):  # noqa: F811
    running(logged_in_api, cluster, metrics=None)


def sample(api, cluster, text, at, name="chat"):
    cluster.metrics[name] = text
    sample_endpoint_stats(api.app.state, at)


def stop_and_restart(api, cluster, at):
    """Stops chat, sampled while stopped, and runs it again."""
    stop(api)
    sample_endpoint_stats(api.app.state, at)
    api.post(api_paths.RESTART_ENDPOINT.format(name="chat"))
    cluster.endpoint_states["chat"] = "running"
    reconcile(api)


def test_the_first_reading_is_only_the_baseline(logged_in_api, chat, cluster, model_registry):
    sample(logged_in_api, cluster, metrics_text(running=2, finished=5), T0)

    assert model_registry.logged("chat") == {}


def test_a_server_sample_holds_gauges_rates_and_the_intervals_latencies(
    logged_in_api, chat, cluster, model_registry
):
    first = metrics_text(finished=10, hits=1, queries=4, ttft=(2, 2, 2), ttft_sum=0.1)
    sample(logged_in_api, cluster, first, T0)
    busy = metrics_text(
        running=2, waiting=1, finished=15, hits=7, queries=12, ttft=(2, 6, 6), ttft_sum=1.3
    )
    sample(logged_in_api, cluster, busy, T0 + timedelta(seconds=10))

    step = int((T0 + timedelta(seconds=10)).timestamp())
    assert model_registry.logged("chat") == {
        "2026-10-08": {
            step: {
                "running": 2,
                "waiting": 1,
                "kv_cache_usage": 0.5,
                "finished_per_second": 0.5,
                # 6 hits of 8 lookups in the interval, not 7 of 12 since start.
                "prefix_cache_hit_rate": 0.75,
                # The interval's 4 requests, all in the (0.1, 0.5] bucket.
                "time_to_first_token.p50": pytest.approx(0.1 + 0.4 * 2 / 4),
                "time_to_first_token.p95": pytest.approx(0.1 + 0.4 * 3.8 / 4),
                "time_to_first_token.mean": pytest.approx(1.2 / 4),
            }
        }
    }


def test_the_captured_vllm_metrics_make_a_sample_with_prefill_and_decode_time(
    logged_in_api, chat, cluster, model_registry
):
    sample(logged_in_api, cluster, IDLE, T0)
    sample(logged_in_api, cluster, SERVED, T0 + timedelta(seconds=10))

    [[logged]] = [list(samples.values()) for samples in model_registry.logged("chat").values()]
    assert logged["generation_tokens_per_second"] == pytest.approx(6.3)
    assert logged["prompt_tokens_per_second"] == pytest.approx(22.5)
    assert logged["preemptions_per_second"] == 0
    assert logged["prefix_cache_hit_rate"] == pytest.approx(128 / 225)
    assert logged["time_to_first_token.p50"] == pytest.approx(0.01625)
    assert {"prefill_time.p50", "decode_time.mean", "queue_time.p95"} <= set(logged)
    assert "acceptance_rate" not in logged


def test_a_counter_that_went_down_counts_from_zero_as_the_pod_restarted(
    logged_in_api, chat, cluster, model_registry
):
    before = metrics_text(finished=100, hits=5, queries=10, ttft=(50, 90, 100), ttft_sum=30)
    sample(logged_in_api, cluster, before, T0)
    # The prefix-cache counters climbed past their old values, but the pod restarted all the same.
    after = metrics_text(running=1, finished=3, hits=15, queries=20, ttft=(0, 3, 3), ttft_sum=0.9)
    sample(logged_in_api, cluster, after, T0 + timedelta(seconds=10))

    [[logged]] = [list(samples.values()) for samples in model_registry.logged("chat").values()]
    assert logged["finished_per_second"] == pytest.approx(0.3)
    assert logged["time_to_first_token.mean"] == pytest.approx(0.3)
    assert logged["prefix_cache_hit_rate"] == pytest.approx(15 / 20)


def test_idle_samples_are_skipped_after_the_first_which_drops_the_charts_to_zero(
    logged_in_api, chat, cluster, model_registry
):
    busy = metrics_text(running=1, finished=4, ttft=(4, 4, 4), ttft_sum=0.2)
    done = metrics_text(finished=4, ttft=(4, 4, 4), ttft_sum=0.2)
    texts = [metrics_text(), metrics_text(), busy, done, done, busy]
    for seconds, text in zip(range(0, 60, 10), texts, strict=True):
        sample(logged_in_api, cluster, text, T0 + timedelta(seconds=seconds))

    logged = model_registry.logged("chat")["2026-10-08"]
    step = int(T0.timestamp())
    # Idle at 10, busy at 20, the first idle at 30 logged, idle at 40 skipped, running at 50.
    assert sorted(logged) == [step + 20, step + 30, step + 50]
    assert logged[step + 30] == {
        "running": 0,
        "waiting": 0,
        "kv_cache_usage": 0,
        "finished_per_second": 0,
    }


def test_the_day_run_changes_at_utc_midnight(logged_in_api, chat, cluster, model_registry):
    midnight = datetime(2026, 10, 9, tzinfo=UTC)
    for seconds, finished in ((-20, 0), (-5, 1), (5, 2)):
        text = metrics_text(running=1, finished=finished)
        sample(logged_in_api, cluster, text, midnight + timedelta(seconds=seconds))

    logged = model_registry.logged("chat")
    assert {day: list(samples) for day, samples in logged.items()} == {
        "2026-10-08": [int(midnight.timestamp()) - 5],
        "2026-10-09": [int(midnight.timestamp()) + 5],
    }


def test_an_unreachable_endpoint_is_skipped_and_the_others_sampled(
    logged_in_api, chat, cluster, model_registry
):
    running(logged_in_api, cluster, name="other", metrics=None)
    for seconds, finished in ((0, 0), (10, 1)):
        cluster.metrics["other"] = metrics_text(running=1, finished=finished)
        sample_endpoint_stats(logged_in_api.app.state, T0 + timedelta(seconds=seconds))

    assert model_registry.logged("chat") == {}
    assert list(model_registry.logged("other")["2026-10-08"]) == [int(T0.timestamp()) + 10]


def test_a_restarted_endpoints_first_reading_is_a_baseline_again(
    logged_in_api, chat, cluster, model_registry
):
    sample(logged_in_api, cluster, metrics_text(running=1, finished=50), T0)
    stop_and_restart(logged_in_api, cluster, T0 + timedelta(seconds=10))
    sample(logged_in_api, cluster, metrics_text(running=1, finished=60), T0 + timedelta(seconds=20))

    assert model_registry.logged("chat") == {}


def history(api, span="live", name="chat"):
    return api.get(api_paths.ENDPOINT_HISTORY.format(name=name), params={"span": span})


def log_recent_and_old_samples(api, cluster):
    """A busy sample 6 days ago and one a minute ago."""
    now = datetime.now(UTC)
    for at in (now - timedelta(days=6), now - timedelta(minutes=1)):
        sample(api, cluster, metrics_text(running=1, ttft=(1, 1, 1), ttft_sum=0.05), at)
        text = metrics_text(running=3, finished=1, ttft=(2, 2, 2), ttft_sum=0.1)
        sample(api, cluster, text, at + timedelta(seconds=10))
        stop_and_restart(api, cluster, at + timedelta(seconds=20))


@pytest.mark.parametrize("span", list(config.ENDPOINT_HISTORY_SPANS))
def test_each_span_reads_its_steps_with_the_points_shared_over_its_day_runs(
    logged_in_api, chat, cluster, model_registry, span
):
    log_recent_and_old_samples(logged_in_api, cluster)

    found = history(logged_in_api, span).json()

    duration = config.ENDPOINT_HISTORY_SPANS[span].total_seconds()
    assert found["end"] - found["start"] == duration
    run_ids, _, steps, max_results = model_registry.history_reads[0]
    assert steps == (found["start"], found["end"])
    assert max_results == config.ENDPOINT_HISTORY_POINTS // len(run_ids)


def test_the_history_holds_each_keys_points_with_latencies_grouped_by_chart(
    logged_in_api, chat, cluster, model_registry
):
    log_recent_and_old_samples(logged_in_api, cluster)

    live = history(logged_in_api, "live").json()
    week = history(logged_in_api, "7d").json()

    assert [value for _, value in live["values"]["running"]] == [3]
    assert [value for _, value in week["values"]["running"]] == [3, 3]
    assert [value for _, value in live["latencies"]["time_to_first_token"]["mean"]] == [
        pytest.approx(0.05)
    ]
    assert set(live["latencies"]["queue_time"]) == {"p50", "p95", "mean"}
    assert live["values"]["acceptance_rate"] == []


def test_a_stopped_endpoint_has_its_history(logged_in_api, chat, cluster, model_registry):
    log_recent_and_old_samples(logged_in_api, cluster)
    stop(logged_in_api)

    response = history(logged_in_api, "7d")

    assert response.status_code == 200
    assert len(response.json()["values"]["running"]) == 2


def test_an_endpoint_without_samples_has_an_empty_history(
    logged_in_api, chat, cluster, model_registry
):
    found = history(logged_in_api).json()

    assert found["values"]["running"] == []
    # The start made the experiment for Request Traces, but no sample made a Run.
    assert model_registry.experiments == {"endpoint/chat": {}}


@pytest.mark.parametrize(
    ("name", "span", "status"), [("unknown", "live", 404), ("chat", "2y", 422)]
)
def test_the_history_route_refuses_unknown_endpoints_and_spans(
    logged_in_api, chat, name, span, status
):
    assert history(logged_in_api, span, name).status_code == status
