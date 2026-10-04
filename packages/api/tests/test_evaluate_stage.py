import json

import pytest

from mlp_api.endpoints.lifecycle import reconcile_endpoints
from mlp_core import api_paths, config

from .test_endpoints import register_adapter
from .test_endpoints import start as start_endpoint
from .test_model_cache import cache_benchmark
from .test_pipeline_request import BASE_MODEL, COMMIT, pipeline_request
from .test_pipelines import submit, submittable, submitted_pipeline  # noqa: F401

# Every test submits or validates a request for the Base Model and the `chat` Dataset.
pytestmark = pytest.mark.usefixtures("submittable")

PINNED_BASE_MODEL = f"hf:{BASE_MODEL}@{COMMIT}"
GSM8K = "lm_eval:gsm8k"
GSM8K_BYTES = config.BENCHMARKS[GSM8K]["size_bytes"]


def evaluating(**evaluate) -> dict:
    """The finetune request, evaluated on GSM8K unless told otherwise."""
    return {**pipeline_request(), "evaluate": {"benchmarks": [GSM8K], **evaluate}}


def evaluate_only(model: str, **evaluate) -> dict:
    return {"name": "qwen-eval", "evaluate": {"model": model, "benchmarks": [GSM8K], **evaluate}}


def validate(api, request):
    return api.post(api_paths.VALIDATE_PIPELINE, json={"request": request})


def errors(api, request) -> list[dict]:
    response = validate(api, request)
    assert response.status_code == 422, response.text
    return response.json()["detail"]


def running_endpoint(api, cluster, name="chat"):
    assert start_endpoint(api, f"hf:{BASE_MODEL}", name=name).status_code == 201
    cluster.endpoint_states[name] = "running"
    reconcile_endpoints(api.app.state.engine, cluster)


def tasks(cluster) -> dict:
    pipeline, _ = submitted_pipeline(cluster)
    return pipeline["components"]["comp-exit-handler-1"]["dag"]["tasks"]


def inputs(task) -> dict:
    return {name: p["runtimeValue"]["constant"] for name, p in task["inputs"]["parameters"].items()}


def container(cluster, step) -> dict:
    pipeline, _ = submitted_pipeline(cluster)
    return pipeline["deploymentSpec"]["executors"][f"exec-{step}"]["container"]


def test_evaluate_defaults_to_the_output_of_finetune(logged_in_api):
    response = validate(logged_in_api, evaluating())

    assert response.status_code == 200, response.text
    assert response.json()["request"]["evaluate"]["model"] == "@finetune"


def test_a_base_model_is_evaluated_without_finetune_pinned_to_a_commit(logged_in_api):
    response = validate(logged_in_api, evaluate_only(f"hf:{BASE_MODEL}"))

    assert response.status_code == 200, response.text
    resolved = response.json()["request"]
    assert resolved["evaluate"]["model"] == PINNED_BASE_MODEL
    assert resolved.get("finetune") is None


def test_a_model_version_resolves_to_its_latest_version(
    logged_in_api, model_registry, object_store
):
    register_adapter(model_registry, object_store, "qwen-sft", PINNED_BASE_MODEL)

    response = validate(logged_in_api, evaluate_only("model:qwen-sft"))

    assert response.status_code == 200, response.text
    assert response.json()["request"]["evaluate"]["model"] == "model:qwen-sft@1"


def test_unknown_benchmarks_are_rejected(logged_in_api):
    [error] = errors(logged_in_api, evaluating(benchmarks=[GSM8K, "lm_eval:nope"]))

    assert error["loc"] == ["evaluate", "benchmarks", 1]


@pytest.mark.parametrize(
    ("model", "message"),
    [
        ("model:missing", "no Registered Model `missing`"),
        ("endpoint:chat", "no Endpoint chat is running"),
        ("@finetune", "`finetune` isn't enabled"),
    ],
)
def test_a_model_that_cannot_be_evaluated_is_rejected(logged_in_api, model, message):
    [error] = errors(logged_in_api, evaluate_only(model))

    assert error["loc"] == ["evaluate", "model"]
    assert message in error["msg"]


def test_an_endpoint_still_pending_is_rejected(logged_in_api):
    assert start_endpoint(logged_in_api, f"hf:{BASE_MODEL}", name="chat").status_code == 201

    [error] = errors(logged_in_api, evaluate_only("endpoint:chat"))

    assert "no Endpoint chat is running" in error["msg"]


def test_serving_options_are_for_evaluates_own_vllm_not_an_endpoint(logged_in_api, cluster):
    running_endpoint(logged_in_api, cluster)
    serving = {"serving": {"max_model_len": 2048}}

    assert validate(logged_in_api, evaluate_only(f"hf:{BASE_MODEL}", **serving)).status_code == 200
    [error] = errors(logged_in_api, evaluate_only("endpoint:chat", **serving))
    assert "serves with its own options" in error["msg"]


def test_evaluate_without_finetune_needs_a_model(logged_in_api):
    request = evaluate_only("unused")
    del request["evaluate"]["model"]

    [error] = errors(logged_in_api, request)

    assert error["loc"] == ["evaluate", "model"]


def test_a_request_without_stages_is_rejected(logged_in_api):
    [error] = errors(logged_in_api, {"name": "nothing"})

    assert "enable at least one" in error["msg"]


def test_validate_lists_each_benchmark_with_its_catalog_size(logged_in_api, hugging_face):
    hugging_face.sizes[BASE_MODEL] = 400

    answer = validate(logged_in_api, evaluating()).json()

    assert answer["downloads"] == [
        {"kind": "base_model", "ref": PINNED_BASE_MODEL, "bytes": 400, "cached": False},
        {"kind": "benchmark", "ref": GSM8K, "bytes": GSM8K_BYTES, "cached": False},
    ]
    assert answer["download_bytes"] == 400 + GSM8K_BYTES


def test_a_fetched_benchmark_is_cached_and_downloads_nothing(logged_in_api, model_cache):
    cache_benchmark(model_cache, GSM8K, 300, days_since_use=1)

    answer = validate(logged_in_api, evaluating()).json()

    [benchmark] = [d for d in answer["downloads"] if d["kind"] == "benchmark"]
    assert benchmark == {"kind": "benchmark", "ref": GSM8K, "bytes": 300, "cached": True}
    assert answer["cached_bytes"] == 300


def test_fetch_downloads_the_base_model_and_the_benchmarks(logged_in_api, cluster):
    submit(logged_in_api, evaluating(benchmarks=[GSM8K, "lm_eval:ifeval"]))

    references = inputs(tasks(cluster)["fetch"])["references"]
    assert references == f"{PINNED_BASE_MODEL},{GSM8K},lm_eval:ifeval"


def test_evaluate_runs_after_finetune_and_before_serve(logged_in_api, cluster):
    pipeline_id = submit(logged_in_api, {**evaluating(), "serve": {}}).json()["id"]

    evaluate = tasks(cluster)["evaluate"]
    assert evaluate["dependentTasks"] == ["finetune"]
    assert tasks(cluster)["serve"]["dependentTasks"] == ["evaluate"]
    assert inputs(evaluate)["pipeline_id"] == str(pipeline_id)
    assert json.loads(inputs(evaluate)["request"])["evaluate"]["model"] == "@finetune"
    assert inputs(evaluate)["endpoint_url"] == ""


def test_evaluate_serves_the_model_with_vllm_on_the_platforms_gpus_offline(logged_in_api, cluster):
    submit(logged_in_api, evaluating())

    step = container(cluster, "evaluate")
    assert step["image"] == "mlp-stages:test"
    assert step["command"] == ["mlp-stage", "evaluate"]
    assert step["resources"]["accelerator"]["resourceCount"] == "1"
    assert inputs(tasks(cluster)["evaluate"])["gpus"] == "1"
    for variable in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
        assert {"name": variable, "value": "1"} in step["env"]
    assert {"name": "HF_HOME", "value": "/model-cache"} in step["env"]


def test_evaluating_a_base_model_fetches_it_first(logged_in_api, cluster):
    submit(logged_in_api, evaluate_only(f"hf:{BASE_MODEL}"))

    assert set(tasks(cluster)) == {"fetch", "evaluate"}
    assert tasks(cluster)["evaluate"]["dependentTasks"] == ["fetch"]
    assert inputs(tasks(cluster)["fetch"])["references"] == f"{PINNED_BASE_MODEL},{GSM8K}"


def test_evaluating_an_adapter_fetches_its_base_model(
    logged_in_api, cluster, model_registry, object_store
):
    register_adapter(model_registry, object_store, "qwen-sft", PINNED_BASE_MODEL)

    submit(logged_in_api, evaluate_only("model:qwen-sft@1"))

    assert inputs(tasks(cluster)["fetch"])["references"] == f"{PINNED_BASE_MODEL},{GSM8K}"


def test_a_running_endpoint_is_evaluated_through_its_service_without_a_gpu(logged_in_api, cluster):
    running_endpoint(logged_in_api, cluster)

    response = submit(logged_in_api, evaluate_only("endpoint:chat"))

    assert response.status_code == 202, response.text
    evaluate = tasks(cluster)["evaluate"]
    assert inputs(evaluate)["endpoint_url"] == "http://endpoint-chat.mlp.svc:8000"
    assert "resources" not in container(cluster, "evaluate")
    assert inputs(evaluate)["gpus"] == "0"
    assert inputs(tasks(cluster)["fetch"])["references"] == GSM8K


def test_evaluate_is_listed_as_a_stage(logged_in_api):
    submit(logged_in_api, evaluate_only(f"hf:{BASE_MODEL}"))

    [pipeline] = logged_in_api.get(api_paths.PIPELINES).json()
    assert pipeline["stages"] == ["evaluate"]


def test_the_benchmark_catalog_lists_category_description_and_size(logged_in_api):
    response = logged_in_api.get(api_paths.BENCHMARKS)

    assert response.status_code == 200
    gsm8k = next(entry for entry in response.json() if entry["name"] == GSM8K)
    assert gsm8k["category"] == "maths"
    assert gsm8k["description"]
    assert gsm8k["size_bytes"] == GSM8K_BYTES
