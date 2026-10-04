import json

import pytest

from mlp_api.endpoints.lifecycle import list_endpoints
from mlp_api.models.mlflow import ModelVersion
from mlp_api.pipelines.reconciler import reconcile_once
from mlp_core import api_paths

from .test_endpoints import contains, vllm_args
from .test_endpoints import start as start_endpoint
from .test_pipeline_request import BASE_MODEL, COMMIT, pipeline_request
from .test_pipelines import (  # noqa: F401
    HF_TOKEN,
    finish_run,
    pipelines,
    submit,
    submittable,
    submitted_pipeline,
)

# Every test submits or validates a request for the Base Model and the `chat` Dataset.
pytestmark = pytest.mark.usefixtures("submittable")

PINNED_BASE_MODEL = f"hf:{BASE_MODEL}@{COMMIT}"


def serving(**serve) -> dict:
    return {**pipeline_request(), "serve": serve}


def validate(api, request) -> dict:
    return api.post(api_paths.VALIDATE_PIPELINE, json={"request": request}).json()


def tasks(cluster) -> dict:
    pipeline, _ = submitted_pipeline(cluster)
    return pipeline["components"]["comp-exit-handler-1"]["dag"]["tasks"]


def test_serve_starts_after_finetune_from_the_stages_image(logged_in_api, cluster):
    pipeline_id = submit(logged_in_api, serving(max_model_len=2048)).json()["id"]

    serve = tasks(cluster)["serve"]
    assert serve["dependentTasks"] == ["finetune"]
    pipeline, kubernetes = submitted_pipeline(cluster)
    container = pipeline["deploymentSpec"]["executors"]["exec-serve"]["container"]
    assert container["image"] == "mlp-stages:test"
    assert container["command"] == ["mlp-stage", "serve"]
    assert {"name": "API_URL", "value": "http://api.mlp.test:8000"} in container["env"]
    assert "resources" not in container
    parameters = serve["inputs"]["parameters"]
    inputs = {name: p["runtimeValue"]["constant"] for name, p in parameters.items()}
    # The API serves from the request it stored, never from what a step sends.
    assert inputs == {"pipeline_id": str(pipeline_id)}
    [secret] = kubernetes["deploymentSpec"]["executors"]["exec-serve"]["secretAsEnv"]
    assert secret["secretName"] == f"pipeline-{pipeline_id}"
    assert secret["keyToEnv"] == [{"secretKey": "step_token", "envVar": "MLP_STEP_TOKEN"}]


def test_a_pipeline_without_serve_has_no_serve_step_nor_step_token(logged_in_api, cluster):
    pipeline_id = submit(logged_in_api, secrets={"hf_token": HF_TOKEN}).json()["id"]

    assert "serve" not in tasks(cluster)
    assert cluster.secrets[f"pipeline-{pipeline_id}"] == {"hf_token": HF_TOKEN}


def test_serve_is_listed_as_a_stage(logged_in_api, cluster):
    submit(logged_in_api, serving())

    assert pipelines(logged_in_api)[0]["stages"] == ["finetune", "serve"]


def test_the_endpoint_is_named_after_the_pipeline_unless_serve_names_it(logged_in_api):
    assert validate(logged_in_api, serving())["request"]["serve"]["name"] == "qwen-sft"
    assert validate(logged_in_api, serving(name="chat"))["request"]["serve"]["name"] == "chat"


def test_a_pipeline_name_that_cannot_name_an_endpoint_needs_a_serve_name(logged_in_api):
    request = {**serving(), "name": "qwen.sft"}

    [error] = validate(logged_in_api, request)["detail"]

    assert error["loc"] == ["serve", "name"]
    assert "can't name an Endpoint" in error["msg"]


def test_a_serve_name_taken_by_a_running_endpoint_is_rejected(logged_in_api):
    start_endpoint(logged_in_api, f"hf:{BASE_MODEL}", name="qwen-sft")

    [error] = validate(logged_in_api, serving())["detail"]

    assert error["loc"] == ["serve", "name"]
    assert "already running" in error["msg"]


def test_serving_options_are_checked_like_an_endpoints(logged_in_api):
    [error] = validate(logged_in_api, serving(gpu_memory_utilization=2))["detail"]

    assert error["loc"] == ["serve", "gpu_memory_utilization"]


# What the `serve` step does: call the API with the step token from the Pipeline's Secret.
def serve_step(api, pipeline_id, token=None):
    """The API's answer to the step, sent without the login cookie the test client holds."""
    secret = api.app.state.cluster.secrets.get(f"pipeline-{pipeline_id}", {})
    token = token or secret["step_token"]
    response = api.post(
        api_paths.SERVE_PIPELINE.format(id=pipeline_id),
        headers={"authorization": f"Bearer {token}"},
    )
    return response


def register_output(model_registry, object_store, pipeline_id, version, name="qwen-sft"):
    """An Adapter the Pipeline's finetune Phase registered on the pinned Base Model."""
    prefix = f"1/run-{name}-{version}/artifacts/model/"
    object_store.buckets["mlflow"][prefix + "adapter_config.json"] = json.dumps({"r": 16})
    tags = {"weights": "adapter", "base_model": PINNED_BASE_MODEL, "pipeline": str(pipeline_id)}
    model_registry.versions.append(ModelVersion(name, version, tags, prefix))


@pytest.fixture
def served_pipeline(logged_in_api, model_registry, object_store) -> int:
    """A submitted Pipeline with `serve`, whose finetune registered version 2; version 3 is
    another Pipeline's."""
    pipeline_id = submit(logged_in_api, serving(max_model_len=2048)).json()["id"]
    register_output(model_registry, object_store, pipeline_id, 2)
    register_output(model_registry, object_store, pipeline_id + 1, 3)
    logged_in_api.cookies.clear()
    return pipeline_id


def test_the_serve_step_starts_an_endpoint_for_the_pipelines_last_model_version(
    logged_in_api, served_pipeline, cluster
):
    response = serve_step(logged_in_api, served_pipeline)

    assert response.status_code == 201, response.text
    started = response.json()
    assert (started["name"], started["model"]) == ("qwen-sft", "model:qwen-sft@2")
    assert (started["status"], started["url"]) == ("pending", "/endpoints/qwen-sft/v1")
    assert contains(vllm_args(cluster, "qwen-sft"), ["--max-model-len", "2048"])


def test_a_step_token_only_serves_its_own_pipeline(logged_in_api, served_pipeline, cluster):
    token = cluster.secrets[f"pipeline-{served_pipeline}"]["step_token"]

    other = serve_step(logged_in_api, served_pipeline + 1, token=token)
    listing = logged_in_api.get(api_paths.ENDPOINTS, headers={"authorization": f"Bearer {token}"})
    verify = logged_in_api.get(api_paths.VERIFY, headers={"authorization": f"Bearer {token}"})

    assert (other.status_code, listing.status_code, verify.status_code) == (401, 401, 401)
    assert cluster.endpoints == {}


def test_a_pipeline_that_registered_nothing_cannot_serve(logged_in_api, cluster, model_registry):
    pipeline_id = submit(logged_in_api, serving()).json()["id"]
    logged_in_api.cookies.clear()

    response = serve_step(logged_in_api, pipeline_id)

    assert response.status_code == 422
    assert "registered no Model Version" in response.json()["detail"]


def test_a_name_taken_meanwhile_fails_the_serve_step(logged_in_api, served_pipeline, cluster):
    serve_step(logged_in_api, served_pipeline)

    response = serve_step(logged_in_api, served_pipeline)

    assert response.status_code == 409


def test_the_endpoint_outlives_its_pipeline(logged_in_api, served_pipeline, cluster):
    serve_step(logged_in_api, served_pipeline)
    finish_run(cluster, "SUCCEEDED")
    cluster.endpoint_states["qwen-sft"] = "running"
    reconcile_once(logged_in_api.app.state)

    assert cluster.secrets == {}
    [endpoint] = list_endpoints(logged_in_api.app.state.engine)
    assert (endpoint["name"], endpoint["status"]) == ("qwen-sft", "running")


def test_a_finished_pipeline_serves_no_more(logged_in_api, served_pipeline, cluster):
    token = cluster.secrets[f"pipeline-{served_pipeline}"]["step_token"]
    finish_run(cluster, "FAILED")
    reconcile_once(logged_in_api.app.state)

    response = serve_step(logged_in_api, served_pipeline, token=token)

    assert response.status_code == 422
    assert cluster.endpoints == {}
