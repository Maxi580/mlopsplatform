import json

import pytest

from mlp_api.pipelines.cluster import KubeflowRun
from mlp_api.pipelines.hugging_face import HubModel
from mlp_api.pipelines.reconciler import reconcile_pipelines
from mlp_api.smoke_tests.lifecycle import clean_up_smoke_tests
from mlp_core import api_paths

from .test_model_uploads import FULL_WEIGHTS
from .test_models import models, register

QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
TINY_QWEN = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
COMMIT = "c0ffee"


@pytest.fixture
def qwen_on_the_hub(logged_in_api, hugging_face):
    for repo in (QWEN, TINY_QWEN):
        hugging_face.models[repo] = HubModel(commit=COMMIT, needs_remote_code=False)
    # A download leaves its own bookkeeping next to the model files.
    hugging_face.files[TINY_QWEN] = {**FULL_WEIGHTS, ".cache/huggingface/download.lock": b""}


def start(api, selection=None):
    if selection is None:
        return api.post(api_paths.SMOKE_TEST_COMPLETE)
    return api.post(api_paths.SMOKE_TEST_CUSTOM, json=selection)


def nodes(cluster) -> list[dict]:
    """The submitted run's tasks in the order they run."""
    tasks = cluster.submitted["pipeline_spec"]["pipeline_spec"]["root"]["dag"]["tasks"]
    ordered, previous = [], None
    while len(ordered) < len(tasks):
        [task] = [t for t in tasks.values() if t.get("dependentTasks", [None])[0] == previous]
        ordered.append(task)
        previous = next(key for key, t in tasks.items() if t is task)
    return ordered


def case_names(cluster) -> list[str]:
    return [task["taskInfo"]["name"] for task in nodes(cluster)]


def smoke_test(api) -> dict:
    [found] = [p for p in api.get(api_paths.PIPELINES).json() if p["cases"] is not None]
    return found


def test_the_complete_smoke_test_runs_fetch_then_every_finetune_case(
    logged_in_api, qwen_on_the_hub, cluster
):
    response = start(logged_in_api)

    assert response.status_code == 202, response.text
    started = response.json()
    assert started["name"].startswith("smoketest-")
    assert started["kubeflow_run_url"] == "/pipeline/#/runs/details/run-1"
    assert case_names(cluster) == ["fetch", "sft-lora-hf", "uploaded-model"]
    assert cluster.submitted["display_name"] == started["name"]


def test_every_case_runs_even_after_the_one_before_it_failed(
    logged_in_api, qwen_on_the_hub, cluster
):
    start(logged_in_api)

    first, *later = nodes(cluster)
    assert "triggerPolicy" not in first
    assert later
    assert all(t["triggerPolicy"]["strategy"] == "ALL_UPSTREAM_TASKS_COMPLETED" for t in later)


@pytest.mark.parametrize(
    ("selection", "cases"),
    [
        ({}, ["fetch"]),
        ({"uploaded_model": True}, ["fetch", "uploaded-model"]),
        ({"finetune": {"phases": ["sft"]}}, ["fetch", "sft-lora-hf"]),
        ({"finetune": {"phases": ["sft"], "methods": [], "backends": ["hf"]}}, ["fetch"]),
    ],
)
def test_a_custom_smoke_test_runs_only_the_selected_cases(
    logged_in_api, qwen_on_the_hub, cluster, selection, cases
):
    response = start(logged_in_api, selection)

    assert response.status_code == 202, response.text
    assert case_names(cluster) == cases
    assert list(smoke_test(logged_in_api)["cases"]) == cases


@pytest.mark.parametrize(
    "selection",
    [{"finetune": {"phases": ["ppo"]}}, {"finetune": {"backends": ["axolotl"]}}, {"quiz": True}],
)
def test_a_selection_of_unknown_cases_is_rejected(
    logged_in_api, qwen_on_the_hub, cluster, selection
):
    response = start(logged_in_api, selection)

    assert response.status_code == 422
    assert cluster.runs == {}


def test_a_finetune_case_trains_the_pinned_base_model_on_its_uploaded_bundled_dataset(
    logged_in_api, qwen_on_the_hub, cluster
):
    name = start(logged_in_api).json()["name"]

    _, finetune, _ = nodes(cluster)
    parameters = finetune["inputs"]["parameters"]
    request = json.loads(parameters["request"]["runtimeValue"]["constant"])
    assert request["name"] == f"{name}-sft-lora-hf"
    assert request["finetune"]["base_model"] == f"hf:{QWEN}@{COMMIT}"
    [phase] = request["finetune"]["phases"]
    assert phase["dataset"] == f"dataset:{name}-sft@1"
    datasets = logged_in_api.get(api_paths.DATASETS).json()
    assert [d["name"] for d in datasets] == [f"{name}-sft"]


def test_the_uploaded_model_case_finetunes_a_tiny_model_uploaded_the_normal_way(
    logged_in_api, qwen_on_the_hub, cluster
):
    name = start(logged_in_api).json()["name"]

    *_, uploaded_model = nodes(cluster)
    parameters = uploaded_model["inputs"]["parameters"]
    request = json.loads(parameters["request"]["runtimeValue"]["constant"])
    assert request["name"] == f"{name}-uploaded-model"
    assert request["finetune"]["from"] == f"model:{name}-uploaded@1"
    [version] = models(logged_in_api)[f"{name}-uploaded"]
    assert (version["tags"]["source"], version["tags"]["weights"]) == ("uploaded", "full")
    assert version["size_bytes"] == sum(len(content) for content in FULL_WEIGHTS.values())


def test_only_one_smoke_test_runs_at_a_time(logged_in_api, qwen_on_the_hub, cluster):
    start(logged_in_api)

    response = start(logged_in_api, {})

    assert response.status_code == 409
    assert len(cluster.runs) == 1


def reconcile(api):
    state = api.app.state
    reconcile_pipelines(state.engine, state.cluster)
    clean_up_smoke_tests(state.engine, state.object_store, state.model_registry)


def test_each_case_reports_its_own_result(logged_in_api, qwen_on_the_hub, cluster):
    start(logged_in_api)
    cluster.runs["run-1"] = KubeflowRun("RUNNING", None, {"fetch": "FAILED"})

    reconcile(logged_in_api)

    assert smoke_test(logged_in_api)["cases"] == {
        "fetch": "failed",
        "sft-lora-hf": "pending",
        "uploaded-model": "pending",
    }


def test_a_failed_case_does_not_stop_the_others(logged_in_api, qwen_on_the_hub, cluster):
    start(logged_in_api)
    cluster.runs["run-1"] = KubeflowRun(
        "FAILED", None, {"fetch": "FAILED", "sft-lora-hf": "SUCCEEDED"}
    )

    reconcile(logged_in_api)

    found = smoke_test(logged_in_api)
    assert found["status"] == "failed"
    assert found["cases"] == {
        "fetch": "failed",
        "sft-lora-hf": "passed",
        "uploaded-model": "failed",
    }


def test_a_case_that_never_ran_in_a_finished_run_failed(logged_in_api, qwen_on_the_hub, cluster):
    start(logged_in_api)
    cluster.runs["run-1"] = KubeflowRun("FAILED", None, {"fetch": "SUCCEEDED"})

    reconcile(logged_in_api)

    assert smoke_test(logged_in_api)["cases"] == {
        "fetch": "passed",
        "sft-lora-hf": "failed",
        "uploaded-model": "failed",
    }


def test_afterwards_only_its_kubeflow_run_remains(
    logged_in_api, qwen_on_the_hub, cluster, model_registry, object_store
):
    name = start(logged_in_api).json()["name"]
    smoke_test_id = smoke_test(logged_in_api)["id"]
    register(model_registry, object_store, f"{name}-sft-lora-hf", 1, pipeline_id=smoke_test_id)
    register(model_registry, object_store, "qwen-sft", 1)
    cluster.runs["run-1"] = KubeflowRun("SUCCEEDED", None, {"fetch": "SUCCEEDED"})

    reconcile(logged_in_api)

    assert logged_in_api.get(api_paths.DATASETS).json() == []
    assert [v.name for v in model_registry.versions] == ["qwen-sft"]
    assert model_registry.deleted_models == [f"{name}-sft-lora-hf", f"{name}-uploaded"]
    assert object_store.objects == {}
    assert list(object_store.buckets["mlflow"]) == [
        "1/run-qwen-sft-1/artifacts/model/adapter_model.safetensors"
    ]
    assert cluster.secrets == {}
    assert (list(cluster.runs), cluster.terminated) == (["run-1"], [])


def test_its_datasets_stay_while_it_runs(logged_in_api, qwen_on_the_hub, cluster):
    start(logged_in_api)
    cluster.runs["run-1"] = KubeflowRun("RUNNING", None)

    reconcile(logged_in_api)

    assert len(logged_in_api.get(api_paths.DATASETS).json()) == 1


def test_a_smoke_test_that_could_not_start_is_failed_and_cleaned_up(
    logged_in_api, hugging_face, cluster
):
    response = start(logged_in_api)

    assert response.status_code == 502
    assert QWEN in response.json()["detail"]
    assert smoke_test(logged_in_api)["status"] == "failed"
    reconcile(logged_in_api)
    assert logged_in_api.get(api_paths.DATASETS).json() == []


def test_smoke_tests_require_login(api, cluster):
    assert start(api).status_code == 401


def test_cancelling_a_smoke_test_fails_its_unfinished_cases(
    logged_in_api, qwen_on_the_hub, cluster
):
    smoke_test_id = start(logged_in_api).json()["id"]
    cluster.runs["run-1"] = KubeflowRun("RUNNING", None, {"fetch": "SUCCEEDED"})
    reconcile(logged_in_api)

    logged_in_api.post(api_paths.CANCEL_PIPELINE.format(id=smoke_test_id))

    assert smoke_test(logged_in_api)["cases"] == {
        "fetch": "passed",
        "sft-lora-hf": "failed",
        "uploaded-model": "failed",
    }
