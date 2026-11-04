from datetime import UTC, datetime

import pytest
from sqlalchemy import update

from mlp_api.endpoints.stats_history import sample_endpoint_stats
from mlp_api.judge.lifecycle import reconcile_judge
from mlp_api.judge.settings import judge as judge_table
from mlp_api.smoke_tests.judge_case import restore_judging
from mlp_core import api_paths, config

from .test_endpoint_stats import SERVED
from .test_endpoints import (
    PINNED_QWEN,
    QWEN,
    endpoints,
    pod,
    qwen_on_the_hub,  # noqa: F401
    reconcile,
    start,
)

ON = {"enabled": True, "spec": {"model": f"hf:{QWEN}"}, "rubric": "Be strict."}


def save(api, **settings):
    return api.put(api_paths.JUDGE, json={**ON, **settings})


def judge(api) -> dict:
    response = api.get(api_paths.JUDGE)
    assert response.status_code == 200, response.text
    return response.json()


def rounds(api):
    """One reconcile round of Endpoints and the Judge."""
    reconcile(api)
    reconcile_judge(api.app.state)


def test_judging_is_off_with_the_default_rubric_until_saved(logged_in_api):
    assert judge(logged_in_api) == {
        "enabled": False,
        "spec": None,
        "rubric": config.JUDGE_RUBRIC,
        "status": None,
        "model": None,
    }


def test_a_judge_stored_with_quantization_still_loads_without_it(
    logged_in_api,
    qwen_on_the_hub,  # noqa: F811
    cluster,
):
    assert save(logged_in_api).status_code == 200
    stored = {"model": f"hf:{QWEN}", "max_model_len": 4096, "quantization": "fp8"}
    with logged_in_api.app.state.engine.begin() as connection:
        connection.execute(update(judge_table).values(spec=stored))

    assert judge(logged_in_api)["spec"] == {"model": f"hf:{QWEN}", "max_model_len": 4096}
    rounds(logged_in_api)


def test_a_smoke_test_restores_judge_settings_saved_with_quantization(
    logged_in_api,
    qwen_on_the_hub,  # noqa: F811
    cluster,
):
    model = f"hf:{QWEN}"
    assert save(logged_in_api, spec={"model": model}).status_code == 200
    spec = {"model": model, "quantization": None}
    restore = {"enabled": False, "spec": spec, "rubric": "Be kind."}

    restore_judging(logged_in_api.app.state, {"model": model, "restore": restore})

    restored = judge(logged_in_api)
    assert (restored["enabled"], restored["spec"]) == (False, {"model": model})


def test_switching_judging_on_starts_the_judge_as_an_endpoint_without_route_or_tracing(
    logged_in_api,
    qwen_on_the_hub,  # noqa: F811
    cluster,
):
    response = save(logged_in_api, spec={"model": f"hf:{QWEN}", "max_model_len": 4096})

    assert response.status_code == 200, response.text
    found = response.json()
    assert (found["enabled"], found["rubric"], found["status"]) == (True, "Be strict.", "pending")
    assert found["model"] == PINNED_QWEN
    manifests = cluster.endpoints["judge"]
    assert "route" not in manifests
    [vllm] = pod(cluster, "judge")["containers"]
    assert not any(arg.startswith("--otlp") for arg in vllm["args"])
    assert "4096" in vllm["args"]
    [listed] = endpoints(logged_in_api)
    assert (listed["name"], listed["judge"]) == ("judge", True)


def test_a_judge_gone_after_a_crash_or_upgrade_is_started_again(
    logged_in_api,
    qwen_on_the_hub,  # noqa: F811
    cluster,
):
    save(logged_in_api)
    cluster.endpoints.clear()

    rounds(logged_in_api)

    assert "judge" in cluster.endpoints
    assert judge(logged_in_api)["status"] == "pending"


def test_a_new_spec_replaces_the_judge_and_a_new_rubric_alone_keeps_it(
    logged_in_api,
    qwen_on_the_hub,  # noqa: F811
    cluster,
):
    save(logged_in_api)
    save(logged_in_api, rubric="Be kind.")
    kept = cluster.deleted_endpoints.copy()

    save(logged_in_api, spec={"model": f"hf:{QWEN}", "max_model_len": 2048})

    assert kept == []
    assert cluster.deleted_endpoints == ["judge"]
    assert "2048" in pod(cluster, "judge")["containers"][0]["args"]


def test_switching_judging_off_removes_the_judge(
    logged_in_api,
    qwen_on_the_hub,  # noqa: F811
    cluster,
):
    save(logged_in_api)

    response = save(logged_in_api, enabled=False)

    assert response.json()["status"] is None
    assert cluster.endpoints == {}
    assert endpoints(logged_in_api) == []
    rounds(logged_in_api)
    assert cluster.endpoints == {}


def test_a_judge_model_that_cant_be_served_leaves_judging_off(logged_in_api, hugging_face):
    response = save(logged_in_api)

    assert response.status_code == 422
    assert judge(logged_in_api)["enabled"] is False


def test_judging_on_needs_a_spec(logged_in_api):
    assert save(logged_in_api, spec=None).status_code == 422


@pytest.mark.parametrize(
    "route",
    [api_paths.STOP_ENDPOINT, api_paths.RESTART_ENDPOINT],
)
def test_the_judge_cant_be_stopped_or_restarted_as_an_endpoint(
    logged_in_api,
    qwen_on_the_hub,  # noqa: F811
    cluster,
    route,
):
    save(logged_in_api)

    assert logged_in_api.post(route.format(name="judge")).status_code == 409
    assert "judge" in cluster.endpoints


def test_the_judge_cant_be_deleted_and_its_name_cant_be_taken(
    logged_in_api,
    qwen_on_the_hub,  # noqa: F811
):
    assert start(logged_in_api, name="judge").status_code == 409
    assert logged_in_api.delete(api_paths.ENDPOINT.format(name="judge")).status_code == 409


def test_the_judge_has_no_stats_history(
    logged_in_api,
    qwen_on_the_hub,  # noqa: F811
    cluster,
    model_registry,
):
    save(logged_in_api)
    cluster.endpoint_states["judge"] = "running"
    cluster.metrics["judge"] = SERVED
    rounds(logged_in_api)

    sample_endpoint_stats(logged_in_api.app.state, datetime.now(UTC))

    assert "endpoint/judge" not in model_registry.experiments
