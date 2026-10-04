import json

import pytest

from mlp_api.pipelines.cluster import KubeflowRun
from mlp_api.pipelines.hugging_face import HubModel
from mlp_api.pipelines.reconciler import reconcile_once
from mlp_core import api_paths

from .test_endpoints import register_adapter
from .test_model_uploads import FULL_WEIGHTS
from .test_models import models, register

QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
TINY_QWEN = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
COMMIT = "c0ffee"
# The cases that run as Kubeflow nodes, without the serving cases' Endpoints.
WITHOUT_SERVING = {"finetune": {}, "uploaded_model": True}
SERVING_CASES = ["serve-base-model", "serve-full-weights", "serve-adapter"]
EVALUATE_CASES = ["evaluate-base-model", "evaluate-adapter"]


@pytest.fixture
def qwen_on_the_hub(logged_in_api, hugging_face):
    for repo in (QWEN, TINY_QWEN):
        hugging_face.models[repo] = HubModel(COMMIT, needs_remote_code=False, model_type="qwen2")
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


def node(cluster, case: str) -> dict:
    [found] = [task for task in nodes(cluster) if task["taskInfo"]["name"] == case]
    return found


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
    assert case_names(cluster) == [
        "fetch",
        "sandbox",
        "sft-lora-hf",
        "uploaded-model",
        "distill-tools-distill",
        "distill-tools",
        "evaluate-base-model",
        "evaluate-adapter",
        "finetune-serve-finetune",
        "finetune-serve",
    ]
    assert cluster.submitted["display_name"] == started["name"]


def test_every_case_runs_even_after_the_one_before_it_failed(
    logged_in_api, qwen_on_the_hub, cluster
):
    start(logged_in_api)

    # Only a step that follows its case's earlier Stage waits for that to pass.
    first, *later, serve = nodes(cluster)
    trains_on_distill = node(cluster, "distill-tools")
    assert "triggerPolicy" not in first
    assert "triggerPolicy" not in serve
    assert "triggerPolicy" not in trains_on_distill
    later.remove(trains_on_distill)
    assert later
    assert all(t["triggerPolicy"]["strategy"] == "ALL_UPSTREAM_TASKS_COMPLETED" for t in later)


@pytest.mark.parametrize(
    ("selection", "cases"),
    [
        ({}, ["fetch"]),
        ({"uploaded_model": True}, ["fetch", "uploaded-model"]),
        ({"sandbox": True}, ["fetch", "sandbox"]),
        ({"evaluate": True}, ["fetch", "evaluate-base-model"]),
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

    finetune = node(cluster, "sft-lora-hf")
    parameters = finetune["inputs"]["parameters"]
    request = json.loads(parameters["request"]["runtimeValue"]["constant"])
    assert request["name"] == f"{name}-sft-lora-hf"
    assert request["finetune"]["base_model"] == f"hf:{QWEN}@{COMMIT}"
    [phase] = request["finetune"]["phases"]
    assert phase["dataset"] == f"dataset:{name}-sft@1"
    datasets = logged_in_api.get(api_paths.DATASETS).json()
    assert [d["name"] for d in datasets] == [f"{name}-distill", f"{name}-sft"]


def test_the_uploaded_model_case_finetunes_a_tiny_model_uploaded_the_normal_way(
    logged_in_api, qwen_on_the_hub, cluster
):
    name = start(logged_in_api).json()["name"]

    uploaded_model = node(cluster, "uploaded-model")
    parameters = uploaded_model["inputs"]["parameters"]
    request = json.loads(parameters["request"]["runtimeValue"]["constant"])
    assert request["name"] == f"{name}-uploaded-model"
    assert request["finetune"]["from"] == f"model:{name}-uploaded@1"
    [version] = models(logged_in_api)[f"{name}-uploaded"]
    assert (version["tags"]["source"], version["tags"]["weights"]) == ("uploaded", "full")
    assert version["size_bytes"] == sum(len(content) for content in FULL_WEIGHTS.values())


def test_the_distill_case_distills_with_the_base_model_and_a_tool_then_trains_on_it(
    logged_in_api, qwen_on_the_hub, cluster
):
    started = start(logged_in_api, {"distill": True}).json()
    name = started["name"]

    assert case_names(cluster) == ["fetch", "distill-tools-distill", "distill-tools"]
    assert list(smoke_test(logged_in_api)["cases"]) == ["fetch", "distill-tools"]
    distill, finetune = node(cluster, "distill-tools-distill"), node(cluster, "distill-tools")
    request = json.loads(distill["inputs"]["parameters"]["request"]["runtimeValue"]["constant"])
    assert request["distill"]["teacher"] == f"hf:{QWEN}@{COMMIT}"
    assert request["distill"]["dataset"] == f"dataset:{name}-distill@1"
    assert request["distill"]["tools"][0]["function"]["name"] == "get_weather"
    assert request["distill"]["serving"]["tool_parser"] == "hermes"
    assert request["finetune"]["phases"][0]["dataset"] == "@distill"
    handoff = finetune["inputs"]["parameters"]["distilled_dataset"]["taskOutputParameter"]
    assert handoff["outputParameterKey"] == "dataset"
    assert set(cluster.secrets[f"pipeline-{started['id']}"]) == {"step_token"}


def test_the_distill_cases_step_registers_its_dataset_under_the_cases_name(
    logged_in_api, qwen_on_the_hub, cluster
):
    started = start(logged_in_api, {"distill": True}).json()
    token = cluster.secrets[f"pipeline-{started['id']}"]["step_token"]
    row = {"prompt": "Hi", "completion": "Hello"}

    response = logged_in_api.post(
        api_paths.DISTILL_PIPELINE.format(id=started["id"]),
        content=json.dumps(row) + "\n",
        headers={"authorization": f"Bearer {token}"},
    )

    assert response.status_code == 201, response.text
    assert response.json() == {"dataset": f"dataset:{started['name']}-distill-tools@1"}


def test_the_sandbox_case_runs_snippets_in_the_sandbox_from_a_pipeline_step(
    logged_in_api, qwen_on_the_hub, cluster
):
    start(logged_in_api, {"sandbox": True})

    _, sandbox = nodes(cluster)
    assert sandbox["triggerPolicy"]["strategy"] == "ALL_UPSTREAM_TASKS_COMPLETED"
    pipeline = cluster.submitted["pipeline_spec"]["pipeline_spec"]
    container = pipeline["deploymentSpec"]["executors"]["exec-sandbox"]["container"]
    assert container["command"] == ["mlp-stage", "check-sandbox"]
    assert {"name": "SANDBOX_URL", "value": "http://sandbox.mlp.test:8090"} in container["env"]
    memory = sandbox["inputs"]["parameters"]["sandbox_memory_mb"]["runtimeValue"]["constant"]
    assert memory == "1024"


def test_the_evaluate_cases_run_one_benchmark_on_the_base_model_and_an_adapter(
    logged_in_api, qwen_on_the_hub, cluster
):
    name = start(logged_in_api).json()["name"]

    fetch = node(cluster, "fetch")["inputs"]["parameters"]["references"]["runtimeValue"]
    assert fetch["constant"] == f"hf:{QWEN}@{COMMIT},lm_eval:truthfulqa_mc2"
    models = {}
    for case in ("evaluate-base-model", "evaluate-adapter"):
        parameters = node(cluster, case)["inputs"]["parameters"]
        request = json.loads(parameters["request"]["runtimeValue"]["constant"])
        assert request["evaluate"]["benchmarks"] == ["lm_eval:truthfulqa_mc2"]
        assert request["evaluate"]["limit"] == 5
        models[case] = request["evaluate"]["model"]
    assert models == {
        "evaluate-base-model": f"hf:{QWEN}@{COMMIT}",
        "evaluate-adapter": f"model:{name}-sft-lora-hf@1",
    }


def test_only_one_smoke_test_runs_at_a_time(logged_in_api, qwen_on_the_hub, cluster):
    start(logged_in_api)

    response = start(logged_in_api, {})

    assert response.status_code == 409
    assert len(cluster.runs) == 1


def reconcile(api):
    reconcile_once(api.app.state)


def test_each_case_reports_its_own_result(logged_in_api, qwen_on_the_hub, cluster):
    start(logged_in_api, WITHOUT_SERVING)
    cluster.runs["run-1"] = KubeflowRun("RUNNING", None, {"fetch": "FAILED"})

    reconcile(logged_in_api)

    assert smoke_test(logged_in_api)["cases"] == {
        "fetch": "failed",
        "sft-lora-hf": "pending",
        "uploaded-model": "pending",
    }


def test_a_failed_case_does_not_stop_the_others(logged_in_api, qwen_on_the_hub, cluster):
    start(logged_in_api, WITHOUT_SERVING)
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
    start(logged_in_api, WITHOUT_SERVING)
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
    name = start(logged_in_api, WITHOUT_SERVING).json()["name"]
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

    assert len(logged_in_api.get(api_paths.DATASETS).json()) == 2


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
    smoke_test_id = start(logged_in_api, WITHOUT_SERVING).json()["id"]
    cluster.runs["run-1"] = KubeflowRun("RUNNING", None, {"fetch": "SUCCEEDED"})
    reconcile(logged_in_api)

    logged_in_api.post(api_paths.CANCEL_PIPELINE.format(id=smoke_test_id))

    assert smoke_test(logged_in_api)["cases"] == {
        "fetch": "passed",
        "sft-lora-hf": "failed",
        "uploaded-model": "failed",
    }


def running_endpoints(api) -> dict[str, str]:
    """Each Endpoint that isn't stopped, by name, with the model it serves."""
    listed = api.get(api_paths.ENDPOINTS).json()
    return {e["name"]: e["model"] for e in listed if e["status"] != "stopped"}


def test_the_complete_smoke_test_serves_the_base_model_and_the_uploaded_full_weights(
    logged_in_api, qwen_on_the_hub, cluster
):
    name = start(logged_in_api).json()["name"]

    reconcile(logged_in_api)

    assert list(smoke_test(logged_in_api)["cases"]) == [
        "fetch",
        "sandbox",
        "sft-lora-hf",
        "uploaded-model",
        "distill-tools",
        "evaluate-base-model",
        "evaluate-adapter",
        "finetune-serve",
        *SERVING_CASES,
    ]
    assert running_endpoints(logged_in_api) == {
        f"{name}-serve-base-model": f"hf:{QWEN}@{COMMIT}",
        f"{name}-serve-full-weights": f"model:{name}-uploaded@1",
    }


def test_a_custom_smoke_test_can_serve_without_finetuning(logged_in_api, qwen_on_the_hub, cluster):
    name = start(logged_in_api, {"serving": True}).json()["name"]

    reconcile(logged_in_api)

    assert list(smoke_test(logged_in_api)["cases"]) == ["fetch", *SERVING_CASES[:2]]
    assert list(running_endpoints(logged_in_api)) == [
        f"{name}-serve-full-weights",
        f"{name}-serve-base-model",
    ]


def test_the_adapter_is_served_once_its_finetune_case_registered_it(
    logged_in_api, qwen_on_the_hub, cluster, model_registry, object_store
):
    name = start(logged_in_api).json()["name"]
    register_adapter(model_registry, object_store, f"{name}-sft-lora-hf", f"hf:{QWEN}@{COMMIT}")
    steps = {"fetch": "SUCCEEDED", "sft-lora-hf": "SUCCEEDED"}
    cluster.runs["run-1"] = KubeflowRun("RUNNING", None, steps)

    reconcile(logged_in_api)

    adapter = running_endpoints(logged_in_api)[f"{name}-serve-adapter"]
    assert adapter == f"model:{name}-sft-lora-hf@1"


def test_the_adapter_case_fails_with_its_finetune_case(logged_in_api, qwen_on_the_hub, cluster):
    name = start(logged_in_api).json()["name"]
    cluster.runs["run-1"] = KubeflowRun("RUNNING", None, {"sft-lora-hf": "FAILED"})

    reconcile(logged_in_api)

    assert smoke_test(logged_in_api)["cases"]["serve-adapter"] == "failed"
    assert f"{name}-serve-adapter" not in running_endpoints(logged_in_api)


def test_a_serving_case_passes_once_vllm_is_ready_and_then_stops_its_endpoint(
    logged_in_api, qwen_on_the_hub, cluster
):
    name = start(logged_in_api).json()["name"]
    reconcile(logged_in_api)
    cluster.endpoint_states[f"{name}-serve-base-model"] = "running"
    cluster.endpoint_states[f"{name}-serve-full-weights"] = "failed"

    reconcile(logged_in_api)

    cases = smoke_test(logged_in_api)["cases"]
    assert (cases["serve-base-model"], cases["serve-full-weights"]) == ("passed", "failed")
    assert cluster.endpoints == {}


def test_a_smoke_test_finishes_once_its_serving_cases_did(
    logged_in_api, qwen_on_the_hub, cluster, model_registry, object_store
):
    name = start(logged_in_api).json()["name"]
    register_adapter(model_registry, object_store, f"{name}-sft-lora-hf", f"hf:{QWEN}@{COMMIT}")
    steps = dict.fromkeys(
        ["fetch", "sandbox", "sft-lora-hf", "uploaded-model", "distill-tools", *EVALUATE_CASES]
        + ["finetune-serve"],
        "SUCCEEDED",
    )
    cluster.runs["run-1"] = KubeflowRun("SUCCEEDED", None, steps)

    reconcile(logged_in_api)
    waiting = smoke_test(logged_in_api)["status"]
    for case in SERVING_CASES:
        cluster.endpoint_states[f"{name}-{case}"] = "running"
    reconcile(logged_in_api)
    reconcile(logged_in_api)

    assert waiting == "running"
    found = smoke_test(logged_in_api)
    assert found["status"] == "succeeded"
    assert set(found["cases"].values()) == {"passed"}
    assert model_registry.versions == []


def test_a_cancelled_smoke_test_stops_its_endpoints(logged_in_api, qwen_on_the_hub, cluster):
    smoke_test_id = start(logged_in_api).json()["id"]
    reconcile(logged_in_api)

    logged_in_api.post(api_paths.CANCEL_PIPELINE.format(id=smoke_test_id))
    reconcile(logged_in_api)

    assert cluster.endpoints == {}
    assert running_endpoints(logged_in_api) == {}


def test_the_serve_stage_case_trains_like_the_first_finetune_case_then_serves_it(
    logged_in_api, qwen_on_the_hub, cluster
):
    started = start(logged_in_api).json()

    *_, finetune, serve = nodes(cluster)
    parameters = finetune["inputs"]["parameters"]
    request = json.loads(parameters["request"]["runtimeValue"]["constant"])
    assert request["name"] == f"{started['name']}-finetune-serve"
    assert request["serve"]["name"] == f"{started['name']}-finetune-serve"
    assert request["finetune"]["phases"][0]["algorithm"] == "sft"
    assert serve["dependentTasks"] == [next(k for k, t in tasks(cluster).items() if t is finetune)]
    assert set(cluster.secrets[f"pipeline-{started['id']}"]) == {"step_token"}


def tasks(cluster) -> dict:
    return cluster.submitted["pipeline_spec"]["pipeline_spec"]["root"]["dag"]["tasks"]


def test_the_serve_stage_case_stops_its_endpoint_once_it_has_a_result(
    logged_in_api, qwen_on_the_hub, cluster
):
    name = start(logged_in_api).json()["name"]
    # What its serve step did.
    endpoint = {"name": f"{name}-finetune-serve", "model": f"hf:{QWEN}"}
    logged_in_api.post(api_paths.ENDPOINTS, json=endpoint)
    cluster.runs["run-1"] = KubeflowRun("RUNNING", None, {"finetune-serve": "SUCCEEDED"})

    reconcile(logged_in_api)

    assert smoke_test(logged_in_api)["cases"]["finetune-serve"] == "passed"
    assert f"{name}-finetune-serve" not in running_endpoints(logged_in_api)


def test_a_smoke_test_without_the_serve_stage_case_has_no_secret(
    logged_in_api, qwen_on_the_hub, cluster
):
    start(logged_in_api, WITHOUT_SERVING)

    assert cluster.secrets == {}
