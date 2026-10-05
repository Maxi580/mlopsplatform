import json
from datetime import timedelta

import pytest

from mlp_api.pipelines.cluster import KubeflowRun
from mlp_api.pipelines.hugging_face import HubModel
from mlp_api.pipelines.reconciler import reconcile_pipelines
from mlp_core import api_paths

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
    assert "learning_rate" in schema["$defs"]["PhaseSettings"]["required"]


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
    return spec["pipeline_spec"], spec["platform_spec"]["platforms"]["kubernetes"]


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
    reconcile_pipelines(api.app.state.engine, api.app.state.cluster)


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


def test_the_list_links_the_kubeflow_run_and_the_mlflow_run(logged_in_api, submittable, cluster):
    submit(logged_in_api)
    finish_run(cluster, "RUNNING", mlflow_run_url="https://mlflow.test/#/runs/abc")

    reconcile(logged_in_api)

    [pipeline] = pipelines(logged_in_api)
    [run_id] = cluster.runs
    assert pipeline["kubeflow_run_url"] == f"/pipeline/#/runs/details/{run_id}"
    assert pipeline["mlflow_run_url"] == "https://mlflow.test/#/runs/abc"


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


def test_pipelines_require_login(api, cluster):
    assert submit(api).status_code == 401
    assert cancel(api, 1).status_code == 401
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


def test_finetune_trains_after_fetch_on_the_backends_trainer_image_with_the_platforms_gpus(
    logged_in_api, submittable, cluster
):
    pipeline_id = submit(logged_in_api).json()["id"]

    pipeline, _ = submitted_pipeline(cluster)
    finetune = pipeline["components"]["comp-exit-handler-1"]["dag"]["tasks"]["finetune"]
    assert finetune["dependentTasks"] == ["fetch"]
    container = pipeline["deploymentSpec"]["executors"]["exec-finetune"]["container"]
    assert container["image"] == "mlp-trainer-hf:test"
    assert container["command"] == ["mlp-stage", "finetune"]
    assert container["resources"]["accelerator"]["resourceType"] == "nvidia.com/gpu"
    assert container["resources"]["accelerator"]["resourceCount"] == "1"
    parameters = finetune["inputs"]["parameters"]
    inputs = {name: p["runtimeValue"]["constant"] for name, p in parameters.items()}
    assert inputs["pipeline_id"] == str(pipeline_id)
    assert inputs["phase_index"] == "0"
    request = json.loads(inputs["request"])
    assert request["finetune"]["base_model"] == f"hf:{BASE_MODEL}@{COMMIT}"
    assert request["finetune"]["phases"][0]["dataset"] == "dataset:chat@1"


def test_an_unsloth_phase_trains_on_its_trainer_image_with_one_gpu(
    logged_in_api, submittable, cluster
):
    app = logged_in_api.app
    app.state.settings = app.state.settings.model_copy(update={"gpus_per_stage": 2})

    assert submit(logged_in_api, pipeline_request(backend="unsloth")).status_code == 202

    pipeline, _ = submitted_pipeline(cluster)
    container = pipeline["deploymentSpec"]["executors"]["exec-finetune"]["container"]
    assert container["image"] == "mlp-trainer-unsloth:test"
    assert container["resources"]["accelerator"]["resourceCount"] == "1"


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


def test_finetune_runs_offline_from_the_model_cache_and_reads_datasets_from_the_object_store(
    logged_in_api, submittable, cluster
):
    submit(logged_in_api, secrets={"hf_token": HF_TOKEN})

    pipeline, kubernetes = submitted_pipeline(cluster)
    env = pipeline["deploymentSpec"]["executors"]["exec-finetune"]["container"]["env"]
    assert {"name": "HF_HUB_OFFLINE", "value": "1"} in env
    assert {"name": "HF_HOME", "value": "/model-cache"} in env
    assert {"name": "S3_ENDPOINT_URL", "value": "http://seaweedfs.test:8333"} in env
    assert {"name": "S3_BUCKET", "value": "platform"} in env
    executor = kubernetes["deploymentSpec"]["executors"]["exec-finetune"]
    [mount] = executor["pvcMount"]
    assert (mount["constant"], mount["mountPath"]) == ("model-cache", "/model-cache")
    [object_store_keys] = executor["secretAsEnv"]
    assert object_store_keys["secretName"] == "mlpipeline-minio-artifact"
    assert {k["envVar"] for k in object_store_keys["keyToEnv"]} == {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    }
