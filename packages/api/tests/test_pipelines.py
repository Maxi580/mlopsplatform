import json
from datetime import timedelta

import pytest

from mlp_api.pipelines.cluster import KubeflowRun
from mlp_api.pipelines.hugging_face import HubModel
from mlp_api.pipelines.reconciler import reconcile_pipelines
from mlp_core import api_paths, config

from .test_datasets import CHAT, jsonl, upload
from .test_models import register
from .test_pipeline_request import BASE_MODEL, COMMIT, pipeline_request, then, without


def validate(api, request):
    return api.post(api_paths.VALIDATE_PIPELINE, json={"request": request})


def test_schema_publishes_the_pipeline_request_and_its_basic_values(logged_in_api):
    response = logged_in_api.get(api_paths.SCHEMA)

    assert response.status_code == 200
    schema = response.json()
    assert set(schema["properties"]) >= {"name", "finetune"}
    assert schema["$defs"]["Sweep"]["properties"]["trials"]["default"] == 10


def test_schema_publishes_when_each_field_applies(logged_in_api):
    schema = logged_in_api.get(api_paths.SCHEMA).json()

    phase = schema["$defs"]["Phase"]["properties"]
    assert phase["teacher"]["applies_if"] == "learns_from_teacher"
    assert phase["rewards"]["applies_if"] == "learns_from_rewards"
    assert phase["lora"]["applies_if"] == "new_adapter"
    assert phase["output"]["applies_if"] == "trains_adapter"
    assert schema["$defs"]["Distill"]["properties"]["api_url"]["applies_if"] == "external_teacher"
    calibration = schema["$defs"]["Quantize"]["properties"]["calibration"]
    assert (calibration["applies_if"], calibration["required_if_applies"]) == ("calibrates", True)
    assert schema["uncalibrated_schemes"] == ["fp8-dynamic"]
    assert schema["adapter_methods"] == ["lora", "qlora"]
    assert schema["algorithms"]["distillation"]["learns_from_teacher"] is True
    assert schema["algorithms"]["grpo"]["learns_from_rewards"] is True


def test_schema_publishes_the_acceleration_benchmarks_results_by_option(logged_in_api):
    benchmark = logged_in_api.get(api_paths.SCHEMA).json()["acceleration_benchmark"]

    assert benchmark["intro"].startswith("We benchmarked these methods on 7 open models")
    results = benchmark["results"]
    assert set(results["quantize.scheme"]) == {"fp8-dynamic", "w4a16-gptq", "nvfp4-gptq"}
    assert results["quantize.scheme"]["nvfp4-gptq"]["rows"][0] == {
        "label": "Speedup, all benchmarks",
        "small": "1.36–1.81×",
        "medium": "1.66–1.97×",
    }
    assert set(results["kv_cache_dtype"]) == {"fp8"}
    assert set(results["speculative.method"]) == {"eagle3"}
    # On-the-fly `quantization` was not measured with any option.
    assert results["quantization"] == {}


def test_schema_publishes_each_algorithms_defaults(logged_in_api):
    algorithms = logged_in_api.get(api_paths.SCHEMA).json()["algorithms"]

    assert algorithms["sft"]["default_settings"]["learning_rate"] == 2e-4
    assert algorithms["dpo"]["default_settings"]["beta"] == 0.1
    assert algorithms["rloo"]["default_lora"]["lora_dropout"] == 0.0
    assert algorithms["grpo"]["objective"] == {"metric": "reward", "goal": "maximize"}
    assert algorithms["sft"]["length_setting"] == "max_length"


def test_validate_returns_the_resolved_request(logged_in_api, hugging_face):
    hugging_face.models[BASE_MODEL] = HubModel(commit=COMMIT, needs_remote_code=False)
    upload(logged_in_api, "chat", jsonl(CHAT))

    response = validate(logged_in_api, pipeline_request())

    assert response.status_code == 200, response.text
    finetune = response.json()["request"]["finetune"]
    assert finetune["base_model"] == f"hf:{BASE_MODEL}@{COMMIT}"
    assert finetune["phases"][0]["dataset"] == "dataset:chat@1"


def test_validate_rejects_with_paths_inside_the_pipeline_request(logged_in_api):
    response = validate(
        logged_in_api, without(pipeline_request(), "finetune", "phases", 0, "dataset")
    )

    assert response.status_code == 422
    loc = ["finetune", "phases", 0, "dataset"]
    assert response.json()["detail"] == [{"loc": loc, "msg": "Field required"}]


def test_validation_requires_login(api, hugging_face):
    assert validate(api, pipeline_request()).status_code == 401


HF_TOKEN = "hf_" + "s3cr3tT0ken" * 4


@pytest.fixture
def submittable(logged_in_api, hugging_face):
    hugging_face.models[BASE_MODEL] = HubModel(commit=COMMIT, needs_remote_code=False)
    upload(logged_in_api, "chat", jsonl(CHAT))


def submit(api, request=None, secrets=None):
    submission = {"request": request or pipeline_request(), "secrets": secrets or {}}
    return api.post(api_paths.PIPELINES, json=submission)


def pipelines(api) -> list[dict]:
    response = api.get(api_paths.PIPELINES)
    assert response.status_code == 200, response.text
    return response.json()


def test_submit_returns_the_pipeline_id_right_away(logged_in_api, submittable):
    response = submit(logged_in_api)

    assert response.status_code == 202, response.text
    pipeline_id = response.json()["id"]
    [pipeline] = pipelines(logged_in_api)
    assert pipeline["id"] == pipeline_id
    assert (pipeline["name"], pipeline["owner"], pipeline["status"]) == (
        "qwen-sft",
        "shared",
        "pending",
    )
    assert pipeline["stages"] == ["finetune"]


def submitted_pipeline(cluster) -> tuple[dict, dict]:
    spec = cluster.submitted["pipeline_spec"]
    return spec["pipeline_spec"], spec["platform_spec"].get("platforms", {}).get("kubernetes", {})


def test_the_pipeline_fetches_the_pinned_base_model_and_always_cleans_up(
    logged_in_api, submittable, cluster
):
    submit(logged_in_api)

    pipeline, _ = submitted_pipeline(cluster)
    fetch = pipeline["components"]["comp-exit-handler-1"]["dag"]["tasks"]["fetch"]
    parameters = fetch["inputs"]["parameters"]
    assert parameters["references"]["runtimeValue"]["constant"] == f"hf:{BASE_MODEL}@{COMMIT}"
    # model_cache_size from the settings ConfigMap, so fetch can check that the download fits.
    assert parameters["model_cache_size"]["runtimeValue"]["constant"] == "200Gi"
    cleanup = pipeline["root"]["dag"]["tasks"]["cleanup"]
    assert cleanup["triggerPolicy"]["strategy"] == "ALL_UPSTREAM_TASKS_COMPLETED"
    executors = pipeline["deploymentSpec"]["executors"]
    assert executors["exec-fetch"]["container"]["command"] == ["mlp-stage", "fetch"]
    assert executors["exec-fetch"]["container"]["image"] == "mlp-stages:test"
    assert executors["exec-cleanup"]["container"]["image"] == "mlp-stages:test"


def test_fetch_writes_into_the_model_cache(logged_in_api, submittable, cluster):
    submit(logged_in_api)

    pipeline, kubernetes = submitted_pipeline(cluster)
    [mount] = kubernetes["deploymentSpec"]["executors"]["exec-fetch"]["pvcMount"]
    assert (mount["constant"], mount["mountPath"]) == ("model-cache", "/model-cache")
    env = pipeline["deploymentSpec"]["executors"]["exec-fetch"]["container"]["env"]
    assert {"name": "HF_HOME", "value": "/model-cache"} in env


def test_only_fetch_gets_the_hf_token_from_the_pipelines_secret(
    logged_in_api, submittable, cluster
):
    pipeline_id = submit(logged_in_api, secrets={"hf_token": HF_TOKEN}).json()["id"]

    assert cluster.secrets == {f"pipeline-{pipeline_id}": {"hf_token": HF_TOKEN}}
    _, kubernetes = submitted_pipeline(cluster)
    executors = kubernetes["deploymentSpec"]["executors"]
    pipeline_secret = f"pipeline-{pipeline_id}"
    assert [
        name
        for name, e in executors.items()
        if any(s["secretName"] == pipeline_secret for s in e.get("secretAsEnv", []))
    ] == ["exec-fetch"]
    [secret] = executors["exec-fetch"]["secretAsEnv"]
    assert secret["keyToEnv"] == [{"secretKey": "hf_token", "envVar": "HF_TOKEN"}]


def test_the_token_is_never_a_pipeline_parameter_nor_stored(
    logged_in_api, submittable, cluster, platform_database
):
    response = submit(logged_in_api, secrets={"hf_token": HF_TOKEN})

    assert HF_TOKEN not in json.dumps(cluster.submitted)
    assert HF_TOKEN not in response.text + json.dumps(pipelines(logged_in_api))
    assert HF_TOKEN.encode() not in platform_database.read_bytes()


def test_a_rejected_submission_starts_nothing(logged_in_api, submittable, cluster):
    response = submit(logged_in_api, without(pipeline_request(), "finetune", "base_model"))

    assert response.status_code == 422
    assert (cluster.secrets, cluster.runs, pipelines(logged_in_api)) == ({}, {}, [])


def test_a_pipeline_kubeflow_refused_is_failed(logged_in_api, submittable, cluster):
    def refuse(display_name, pipeline_spec):
        raise ConnectionError("ml-pipeline unreachable")

    cluster.submit_run = refuse

    response = submit(logged_in_api)

    assert response.status_code == 502
    assert "ml-pipeline unreachable" in response.json()["detail"]
    assert pipelines(logged_in_api)[0]["status"] == "failed"


def pipeline(api, pipeline_id):
    return api.get(api_paths.PIPELINE.format(id=pipeline_id))


def test_a_pipeline_returns_its_resolved_request_without_its_secrets(logged_in_api, submittable):
    pipeline_id = submit(logged_in_api, secrets={"hf_token": HF_TOKEN}).json()["id"]

    response = pipeline(logged_in_api, pipeline_id)

    assert response.status_code == 200, response.text
    assert response.json()["id"] == pipeline_id
    finetune = response.json()["request"]["finetune"]
    assert finetune["base_model"] == f"hf:{BASE_MODEL}@{COMMIT}"
    assert finetune["phases"][0]["dataset"] == "dataset:chat@1"
    assert HF_TOKEN not in response.text


def test_a_resolved_request_submits_again_as_a_new_pipeline(logged_in_api, submittable, cluster):
    first = submit(logged_in_api, secrets={"hf_token": HF_TOKEN}).json()["id"]
    request = pipeline(logged_in_api, first).json()["request"]

    response = submit(logged_in_api, request, secrets={"hf_token": "hf_fresh"})

    assert response.status_code == 202, response.text
    second = response.json()["id"]
    assert second != first
    assert pipeline(logged_in_api, second).json()["request"] == request
    assert cluster.secrets[f"pipeline-{second}"] == {"hf_token": "hf_fresh"}


def test_an_unknown_pipeline_is_not_found(logged_in_api):
    assert pipeline(logged_in_api, 42).status_code == 404


def reconcile(api):
    reconcile_pipelines(api.app.state)


def finish_run(cluster, state, mlflow_run_url=None):
    [run_id] = cluster.runs
    cluster.runs[run_id] = KubeflowRun(state, mlflow_run_url)


@pytest.mark.parametrize(
    ("state", "status"),
    [("SUCCEEDED", "succeeded"), ("FAILED", "failed"), ("CANCELED", "cancelled")],
)
def test_a_finished_run_finishes_the_pipeline_and_deletes_its_secret(
    logged_in_api, submittable, cluster, state, status
):
    submit(logged_in_api, secrets={"hf_token": HF_TOKEN})
    finish_run(cluster, state)

    reconcile(logged_in_api)

    assert pipelines(logged_in_api)[0]["status"] == status
    assert cluster.secrets == {}


def test_a_running_pipeline_keeps_its_secret(logged_in_api, submittable, cluster):
    submit(logged_in_api, secrets={"hf_token": HF_TOKEN})
    finish_run(cluster, "RUNNING")

    reconcile(logged_in_api)

    assert pipelines(logged_in_api)[0]["status"] == "running"
    assert len(cluster.secrets) == 1


def test_a_pipeline_whose_pod_waits_for_a_gpu_shows_it(logged_in_api, submittable, cluster):
    submit(logged_in_api)
    finish_run(cluster, "RUNNING")
    cluster.waiting_for_gpu = set(cluster.runs)

    reconcile(logged_in_api)

    assert pipelines(logged_in_api)[0]["status"] == "waiting for GPU"


def test_a_pipeline_whose_trainjob_waits_for_gpus_shows_it(logged_in_api, submittable, cluster):
    pipeline_id = submit(logged_in_api).json()["id"]
    finish_run(cluster, "RUNNING")
    cluster.queued_trainjobs = {pipeline_id}

    reconcile(logged_in_api)

    assert pipelines(logged_in_api)[0]["status"] == "waiting for GPU"


def test_trainjobs_of_finished_or_unknown_pipelines_are_deleted(
    logged_in_api, submittable, cluster
):
    finished, running = (submit(logged_in_api).json()["id"] for _ in range(2))
    run_ids = list(cluster.runs)
    cluster.runs[run_ids[0]] = KubeflowRun("FAILED", None)
    cluster.runs[run_ids[1]] = KubeflowRun("RUNNING", None)
    cluster.trainjobs = {finished, running, 999}

    reconcile(logged_in_api)

    assert cluster.trainjobs == {running}


def test_a_run_deleted_in_the_kfp_ui_fails_the_pipeline(logged_in_api, submittable, cluster):
    submit(logged_in_api)
    cluster.runs.clear()

    reconcile(logged_in_api)

    assert pipelines(logged_in_api)[0]["status"] == "failed"
    assert cluster.secrets == {}


def test_secrets_older_than_48_hours_or_of_no_running_pipeline_are_swept(
    logged_in_api, submittable, cluster
):
    old, fresh = (submit(logged_in_api).json()["id"] for _ in range(2))
    cluster.secret_created[f"pipeline-{old}"] -= timedelta(hours=49)
    cluster.create_secret("pipeline-999", {})

    reconcile(logged_in_api)

    assert list(cluster.secrets) == [f"pipeline-{fresh}"]


MLFLOW_RUN_URL = "https://platform.test/mlflow/#/experiments/1/runs/abc"
REQUEST_ARTIFACT = "1/abc/artifacts/pipeline_request.json"


@pytest.fixture
def mlflow_run(model_registry):
    """The MLflow Run KFP's plugin made for the Pipeline, at MLFLOW_RUN_URL."""
    model_registry.runs["abc"] = "1/abc/artifacts/"


def test_the_list_links_the_kubeflow_run_and_the_mlflow_run(
    logged_in_api, submittable, cluster, mlflow_run
):
    submit(logged_in_api)
    finish_run(cluster, "RUNNING", mlflow_run_url=MLFLOW_RUN_URL)

    reconcile(logged_in_api)

    [pipeline] = pipelines(logged_in_api)
    [run_id] = cluster.runs
    assert pipeline["kubeflow_run_url"] == f"/pipeline/#/runs/details/{run_id}"
    assert pipeline["mlflow_run_url"] == MLFLOW_RUN_URL


@pytest.mark.parametrize("state", ["RUNNING", "SUCCEEDED"])
def test_the_mlflow_run_holds_the_pipelines_request(
    logged_in_api, submittable, cluster, object_store, mlflow_run, state
):
    pipeline_id = submit(logged_in_api, secrets={"hf_token": HF_TOKEN}).json()["id"]
    finish_run(cluster, state, mlflow_run_url=MLFLOW_RUN_URL)

    reconcile(logged_in_api)

    logged = json.loads(object_store.read("mlflow", REQUEST_ARTIFACT))
    assert logged == pipeline(logged_in_api, pipeline_id).json()["request"]
    assert HF_TOKEN not in json.dumps(logged)


def test_a_pipeline_that_finished_keeps_its_mlflow_link_though_the_request_failed(
    logged_in_api, submittable, cluster, object_store
):
    submit(logged_in_api)
    finish_run(cluster, "SUCCEEDED", mlflow_run_url=MLFLOW_RUN_URL)

    reconcile(logged_in_api)

    assert pipelines(logged_in_api)[0]["mlflow_run_url"] == MLFLOW_RUN_URL
    assert REQUEST_ARTIFACT not in object_store.buckets["mlflow"]


def test_the_mlflow_link_waits_until_the_run_holds_the_request(
    logged_in_api, submittable, cluster, model_registry, object_store
):
    submit(logged_in_api)
    finish_run(cluster, "RUNNING", mlflow_run_url=MLFLOW_RUN_URL)

    reconcile(logged_in_api)
    linked_before = pipelines(logged_in_api)[0]["mlflow_run_url"]
    model_registry.runs["abc"] = "1/abc/artifacts/"
    reconcile(logged_in_api)

    assert linked_before is None
    assert pipelines(logged_in_api)[0]["mlflow_run_url"] == MLFLOW_RUN_URL
    assert REQUEST_ARTIFACT in object_store.buckets["mlflow"]


def cancel(api, pipeline_id):
    return api.post(api_paths.CANCEL_PIPELINE.format(id=pipeline_id))


def test_cancel_terminates_the_run_deletes_the_secret_and_marks_it_cancelled(
    logged_in_api, submittable, cluster
):
    pipeline_id = submit(logged_in_api, secrets={"hf_token": HF_TOKEN}).json()["id"]
    finish_run(cluster, "RUNNING")

    response = cancel(logged_in_api, pipeline_id)
    reconcile(logged_in_api)

    assert response.status_code == 200, response.text
    assert cluster.terminated == list(cluster.runs)
    assert cluster.secrets == {}
    assert pipelines(logged_in_api)[0]["status"] == "cancelled"


def test_a_finished_pipeline_cannot_be_cancelled(logged_in_api, submittable, cluster):
    pipeline_id = submit(logged_in_api).json()["id"]
    finish_run(cluster, "SUCCEEDED")
    reconcile(logged_in_api)

    response = cancel(logged_in_api, pipeline_id)

    assert response.status_code == 409
    assert "succeeded" in response.json()["detail"]
    assert cluster.terminated == []


def test_cancelling_an_unknown_pipeline_is_not_found(logged_in_api):
    assert cancel(logged_in_api, 42).status_code == 404


def delete(api, pipeline_id):
    return api.delete(api_paths.PIPELINE.format(id=pipeline_id))


@pytest.mark.parametrize("state", ["SUCCEEDED", "FAILED"])
def test_delete_removes_a_finished_pipeline_and_its_run(logged_in_api, submittable, cluster, state):
    pipeline_id = submit(logged_in_api).json()["id"]
    [run_id] = cluster.runs
    finish_run(cluster, state)
    reconcile(logged_in_api)

    response = delete(logged_in_api, pipeline_id)

    assert response.status_code == 204, response.text
    assert pipelines(logged_in_api) == []
    assert cluster.deleted_runs == [run_id]


def test_delete_removes_the_request_from_the_mlflow_run_and_keeps_the_rest(
    logged_in_api, submittable, cluster, object_store, mlflow_run
):
    pipeline_id = submit(logged_in_api).json()["id"]
    finish_run(cluster, "SUCCEEDED", mlflow_run_url=MLFLOW_RUN_URL)
    reconcile(logged_in_api)
    weights = {"1/def/artifacts/model/adapter_model.safetensors", "1/def/artifacts/model/x.json"}
    for key in weights:
        object_store.buckets["mlflow"][key] = b"kept"

    response = delete(logged_in_api, pipeline_id)

    assert response.status_code == 204, response.text
    assert set(object_store.buckets["mlflow"]) == weights


def test_a_cancelled_pipeline_can_be_deleted(logged_in_api, submittable, cluster):
    pipeline_id = submit(logged_in_api).json()["id"]
    cancel(logged_in_api, pipeline_id)

    assert delete(logged_in_api, pipeline_id).status_code == 204
    assert pipelines(logged_in_api) == []


def test_a_running_pipeline_cannot_be_deleted(logged_in_api, submittable, cluster):
    pipeline_id = submit(logged_in_api).json()["id"]
    finish_run(cluster, "RUNNING")
    reconcile(logged_in_api)

    response = delete(logged_in_api, pipeline_id)

    assert response.status_code == 409
    assert "running" in response.json()["detail"]
    assert cluster.deleted_runs == []
    assert len(pipelines(logged_in_api)) == 1


def test_deleting_an_unknown_pipeline_is_not_found(logged_in_api):
    assert delete(logged_in_api, 42).status_code == 404


def test_pipelines_require_login(api, cluster):
    assert submit(api).status_code == 401
    assert cancel(api, 1).status_code == 401
    assert delete(api, 1).status_code == 401
    assert pipeline(api, 1).status_code == 401


def test_a_cancel_during_a_reconcile_stays_cancelled(logged_in_api, submittable, cluster):
    pipeline_id = submit(logged_in_api).json()["id"]
    finish_run(cluster, "RUNNING")
    find_run = cluster.find_run

    def cancel_meanwhile(run_id):
        run = find_run(run_id)
        cancel(logged_in_api, pipeline_id)
        return run

    cluster.find_run = cancel_meanwhile

    reconcile(logged_in_api)

    assert pipelines(logged_in_api)[0]["status"] == "cancelled"


def trainjob(pipeline: dict, task: str = "finetune") -> dict:
    """The TrainJob the Phase's launcher step creates."""
    parameters = pipeline["components"]["comp-exit-handler-1"]["dag"]["tasks"][task]["inputs"]
    return json.loads(parameters["parameters"]["trainjob"]["runtimeValue"]["constant"])


def test_finetune_launches_after_fetch_a_trainjob_on_the_backends_trainer_image(
    logged_in_api, submittable, cluster
):
    pipeline_id = submit(logged_in_api).json()["id"]

    pipeline, _ = submitted_pipeline(cluster)
    finetune = pipeline["components"]["comp-exit-handler-1"]["dag"]["tasks"]["finetune"]
    assert finetune["dependentTasks"] == ["fetch"]
    launcher = pipeline["deploymentSpec"]["executors"]["exec-finetune"]["container"]
    assert launcher["image"] == "mlp-launcher:test"
    assert launcher["command"] == ["mlp-stage", "launch-phase"]
    assert "accelerator" not in launcher.get("resources", {})
    job = trainjob(pipeline)
    assert job["metadata"]["generateName"] == f"pipeline-{pipeline_id}-phase-0-"
    assert job["metadata"]["labels"] == {
        "mlp/pipeline-id": str(pipeline_id),
        "kueue.x-k8s.io/queue-name": "gpu",
    }
    assert job["spec"]["runtimeRef"] == {"name": "mlp-finetune"}
    trainer = job["spec"]["trainer"]
    assert trainer["image"] == "mlp-trainer-hf:test"
    assert trainer["numNodes"] == 1
    assert trainer["resourcesPerNode"] == {"limits": {"nvidia.com/gpu": 1}}
    parameters = finetune["inputs"]["parameters"]
    inputs = {name: p["runtimeValue"]["constant"] for name, p in parameters.items()}
    assert inputs["pipeline_id"] == str(pipeline_id)
    assert inputs["phase_index"] == "0"
    request = json.loads(inputs["request"])
    assert request["finetune"]["base_model"] == f"hf:{BASE_MODEL}@{COMMIT}"
    assert request["finetune"]["phases"][0]["dataset"] == "dataset:chat@1"


@pytest.mark.parametrize(
    ("method", "nodes", "gpus", "accelerate_config"),
    [
        ("lora", 1, 1, "ddp"),
        ("qlora", 1, 2, "ddp"),
        ("full", 1, 1, "fsdp"),
        ("full", 2, 2, "fsdp"),
    ],
)
def test_a_phase_launches_one_process_per_gpu_of_each_node_with_its_weight_methods_config(
    logged_in_api, submittable, cluster, method, nodes, gpus, accelerate_config
):
    update = {"node_count": nodes, "nodes_per_stage": nodes, "gpus_per_stage": gpus}
    app = logged_in_api.app
    app.state.settings = app.state.settings.model_copy(update={**update, "gpu_count": 4})

    request = pipeline_request(phase={"method": method})
    if method == "full":
        request = without(request, "finetune", "phases", 0, "lora")

    assert submit(logged_in_api, request).status_code == 202

    pipeline, _ = submitted_pipeline(cluster)
    trainer = trainjob(pipeline)["spec"]["trainer"]
    assert trainer["command"] == [
        "accelerate",
        "launch",
        "--config_file",
        f"/app/packages/stages/src/mlp_stages/finetune/accelerate/{accelerate_config}.yaml",
        "--num_processes",
        str(nodes * gpus),
        # Kubeflow Trainer sets these on every node, and Kubernetes fills them in.
        "--num_machines",
        "$(PET_NNODES)",
        "--machine_rank",
        "$(PET_NODE_RANK)",
        "--main_process_ip",
        "$(PET_MASTER_ADDR)",
        "--main_process_port",
        "$(PET_MASTER_PORT)",
        "-m",
        "mlp_stages.main",
        "finetune",
    ]
    assert trainer["numNodes"] == nodes
    assert trainer["resourcesPerNode"] == {"limits": {"nvidia.com/gpu": gpus}}


def test_an_unsloth_phase_trains_on_its_trainer_image_on_one_gpu_of_one_node(
    logged_in_api, submittable, cluster
):
    update = {"node_count": 2, "nodes_per_stage": 2, "gpus_per_stage": 2, "gpu_count": 4}
    app = logged_in_api.app
    app.state.settings = app.state.settings.model_copy(update=update)

    assert submit(logged_in_api, pipeline_request(backend="unsloth")).status_code == 202

    pipeline, _ = submitted_pipeline(cluster)
    trainer = trainjob(pipeline)["spec"]["trainer"]
    assert trainer["image"] == "mlp-trainer-unsloth:test"
    assert (trainer["numNodes"], trainer["resourcesPerNode"]) == (
        1,
        {"limits": {"nvidia.com/gpu": 1}},
    )


def test_each_node_fetches_the_pipelines_base_models_it_is_missing(
    logged_in_api, submittable, cluster
):
    pipeline_id = submit(logged_in_api, secrets={"hf_token": HF_TOKEN}).json()["id"]

    pipeline, _ = submitted_pipeline(cluster)
    [patch] = trainjob(pipeline)["spec"]["runtimePatches"]
    [node] = patch["trainingRuntimeSpec"]["template"]["spec"]["replicatedJobs"]
    [fetch] = node["template"]["spec"]["template"]["spec"]["initContainers"]
    assert fetch["name"] == "fetch"
    assert fetch["env"] == [
        {"name": "BASE_MODELS", "value": f"hf:{BASE_MODEL}@{COMMIT}"},
        {
            "name": "HF_TOKEN",
            "valueFrom": {
                "secretKeyRef": {
                    "name": f"pipeline-{pipeline_id}",
                    "key": "hf_token",
                    "optional": True,
                }
            },
        },
    ]


def test_each_phase_trains_in_its_own_step_from_the_model_version_the_one_before_registered(
    logged_in_api, submittable, cluster
):
    upload(logged_in_api, "pairs", jsonl({"prompt": "Hi", "chosen": "Hello!", "rejected": "Go."}))
    request = pipeline_request()
    request["finetune"]["phases"].append(then("dpo", "dataset:pairs"))

    assert submit(logged_in_api, request).status_code == 202

    pipeline, _ = submitted_pipeline(cluster)
    tasks = pipeline["components"]["comp-exit-handler-1"]["dag"]["tasks"]
    assert set(tasks) == {"fetch", "finetune", "finetune-2"}
    sft, dpo = tasks["finetune"], tasks["finetune-2"]
    assert (sft["taskInfo"]["name"], dpo["taskInfo"]["name"]) == ("finetune-sft", "finetune-dpo")
    assert dpo["dependentTasks"] == ["finetune"]
    first, second = sft["inputs"]["parameters"], dpo["inputs"]["parameters"]
    assert first["phase_index"]["runtimeValue"]["constant"] == "0"
    assert first["previous_model_version"]["runtimeValue"]["constant"] == ""
    assert second["phase_index"]["runtimeValue"]["constant"] == "1"
    handoff = second["previous_model_version"]["taskOutputParameter"]
    assert handoff == {"producerTask": "finetune", "outputParameterKey": "model_version"}


def test_a_pipeline_starting_from_a_model_version_fetches_nothing(
    logged_in_api, submittable, cluster, model_registry, object_store
):
    register(model_registry, object_store, "uploaded", 1, weights="full")
    request = without(pipeline_request(), "finetune", "base_model")
    request["finetune"]["from"] = "model:uploaded"

    assert submit(logged_in_api, request).status_code == 202

    pipeline, _ = submitted_pipeline(cluster)
    tasks = pipeline["components"]["comp-exit-handler-1"]["dag"]["tasks"]
    assert list(tasks) == ["finetune"]
    parameters = tasks["finetune"]["inputs"]["parameters"]
    request = json.loads(parameters["request"]["runtimeValue"]["constant"])
    assert request["finetune"]["from"] == "model:uploaded@1"


@pytest.mark.parametrize("algorithm", list(config.ALGORITHMS))
def test_schema_publishes_every_allowed_setting_once_shown_directly_or_one_click_away(
    logged_in_api, algorithm
):
    schema = logged_in_api.get(api_paths.SCHEMA).json()
    published = schema["algorithms"][algorithm]

    shown, more = published["settings"], published["more_settings"]
    blocked = config.ALGORITHMS[algorithm]["blocked_settings"]
    assert published["length_setting"] in shown
    assert not set(shown) & set(more)
    assert not set(blocked) & (set(shown) | set(more))
    assert {"learning_rate", "bf16", "seed"} <= set(shown)
    assert "adam_beta1" in more
    lora = {*published["lora_settings"], *schema["more_lora_settings"]}
    left_out = {"use_dora", "task_type", "modules_to_save", "runtime_config", "velora_config"}
    assert not left_out & lora
    assert published["lora_settings"]["r"]["enum"] == list(config.VLLM_LORA_RANKS)


def test_schema_publishes_each_row_format_with_an_example(logged_in_api):
    schema = logged_in_api.get(api_paths.SCHEMA).json()

    formats = schema["row_formats"]
    assert set(formats) == set(config.ROW_FORMATS)
    assert json.loads(formats["preference"]["example"]).keys() >= {"chosen", "rejected"}
    assert schema["algorithms"]["dpo"]["row_formats"] == ["preference"]
    distill = schema["$defs"]["Distill"]["properties"]["dataset"]
    assert distill["row_formats"] == ["prompt_only"]
    assert schema["$defs"]["Calibration"]["properties"]["dataset"]["row_formats"] == [
        "messages",
        "text",
    ]


def test_schema_publishes_the_serving_options_and_speculative_decoding(logged_in_api):
    defs = logged_in_api.get(api_paths.SCHEMA).json()["$defs"]

    serving = defs["ServingOptions"]["properties"]
    assert serving["gpu_memory_utilization"]["default"] == 0.9
    assert "description" in serving["max_model_len"]
    speculative = defs["Speculative"]["properties"]
    assert speculative["num_speculative_tokens"]["default"] == 3
    assert speculative["num_speculative_tokens"]["exclusiveMinimum"] == 0


def test_only_gpu_steps_wait_in_the_gpu_queue(logged_in_api, submittable, cluster):
    request = {**pipeline_request(), "evaluate": {"benchmarks": ["lm_eval:gsm8k"]}}
    response = submit(logged_in_api, request)

    assert response.status_code == 202, response.text

    _, platform = submitted_pipeline(cluster)
    executors = platform["deploymentSpec"]["executors"]
    queued = {
        name
        for name, executor in executors.items()
        if executor.get("podMetadata", {}).get("labels", {}).get("kueue.x-k8s.io/queue-name")
    }
    # finetune's TrainJob waits in the queue instead.
    assert queued == {"exec-evaluate"}
