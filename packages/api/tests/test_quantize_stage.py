import gzip
import json

import pytest

from mlp_api.datasets.calibration import register_calibration_dataset
from mlp_core import api_paths, config

from .test_datasets import CHAT, datasets, jsonl, upload
from .test_endpoints import register_adapter
from .test_evaluate_stage import GSM8K, PINNED_BASE_MODEL, container, errors, tasks, validate
from .test_models import register
from .test_pipeline_request import BASE_MODEL, pipeline_request
from .test_pipelines import submit, submittable  # noqa: F401
from .test_resume import failed_pipeline, parameters, registered_phase, resume

# Every test submits or validates a request for the Base Model and the `chat` Dataset.
pytestmark = pytest.mark.usefixtures("submittable")

CALIBRATED_SCHEMES = ["w4a16-gptq", "w4a16-awq", "w8a8-int8"]
QUANTIZED_CONFIG = json.dumps({"quantization_config": {"quant_method": "compressed-tensors"}})


@pytest.fixture
def default_calibration(logged_in_api):
    upload(logged_in_api, config.CALIBRATION_DATASET, jsonl(CHAT))


def quantizing(**quantize) -> dict:
    """The finetune request, quantized to FP8 unless told otherwise."""
    return {**pipeline_request(), "quantize": {"scheme": "fp8-dynamic", **quantize}}


def quantize_only(model: str, **quantize) -> dict:
    return {"name": "qwen-fp8", "quantize": {"model": model, "scheme": "fp8-dynamic", **quantize}}


def resolved(api, request) -> dict:
    response = validate(api, request)
    assert response.status_code == 200, response.text
    return response.json()["request"]


def test_quantize_defaults_to_the_output_of_finetune(logged_in_api):
    assert resolved(logged_in_api, quantizing())["quantize"]["model"] == "@finetune"


def test_a_base_model_is_quantized_without_finetune_pinned_to_a_commit(logged_in_api):
    quantize = resolved(logged_in_api, quantize_only(f"hf:{BASE_MODEL}"))["quantize"]

    assert quantize == {
        "model": PINNED_BASE_MODEL,
        "scheme": "fp8-dynamic",
        "ignore": ["lm_head"],
        "calibration": None,
    }


@pytest.mark.parametrize("scheme", CALIBRATED_SCHEMES)
def test_a_calibrated_scheme_without_calibration_data_is_rejected(logged_in_api, scheme):
    [error] = errors(logged_in_api, quantizing(scheme=scheme))

    assert error["loc"] == ["quantize"]
    assert "add `calibration`" in error["msg"]


def test_fp8_dynamic_needs_no_calibration_data(logged_in_api, default_calibration):
    [error] = errors(logged_in_api, quantizing(calibration={}))

    assert "needs no `calibration`" in error["msg"]


@pytest.mark.parametrize("scheme", CALIBRATED_SCHEMES)
def test_calibration_defaults_to_the_dataset_the_install_registered(
    logged_in_api, default_calibration, scheme
):
    quantize = resolved(logged_in_api, quantizing(scheme=scheme, calibration={}))["quantize"]

    assert quantize["calibration"] == {
        "dataset": f"dataset:{config.CALIBRATION_DATASET}@1",
        "samples": 512,
        "max_length": 2048,
    }


def test_calibration_without_the_default_dataset_needs_one_named(logged_in_api):
    [error] = errors(logged_in_api, quantizing(scheme="w4a16-gptq", calibration={}))

    assert error["loc"] == ["quantize", "calibration", "dataset"]
    assert f"no Dataset `{config.CALIBRATION_DATASET}`" in error["msg"]


def test_calibration_reads_messages_or_text_rows(logged_in_api):
    upload(logged_in_api, "essays", jsonl({"text": "Once upon a time"}))
    upload(logged_in_api, "pairs", jsonl({"prompt": "Hi", "chosen": "Hello", "rejected": "Go"}))
    calibration = {"samples": 64, "max_length": 512}

    text = quantizing(scheme="w4a16-gptq", calibration={"dataset": "dataset:essays", **calibration})
    pairs = quantizing(scheme="w4a16-gptq", calibration={"dataset": "dataset:pairs", **calibration})

    assert resolved(logged_in_api, text)["quantize"]["calibration"]["dataset"] == "dataset:essays@1"
    [error] = errors(logged_in_api, pairs)
    assert "calibrates on messages, text rows" in error["msg"]


def test_a_quantized_base_model_is_rejected(logged_in_api, hugging_face):
    hugging_face.files[BASE_MODEL] = {"config.json": QUANTIZED_CONFIG.encode()}

    [error] = errors(logged_in_api, quantize_only(f"hf:{BASE_MODEL}"))

    assert error["loc"] == ["quantize", "model"]
    assert "already quantized" in error["msg"]


def test_a_quantized_model_version_is_rejected(logged_in_api, model_registry, object_store):
    files = {"config.json": QUANTIZED_CONFIG}
    register(model_registry, object_store, "qwen-fp8", 1, files=files, weights="full")

    [error] = errors(logged_in_api, quantize_only("model:qwen-fp8"))

    assert "already quantized" in error["msg"]


def test_a_quantized_model_version_is_no_finetune_starting_point(
    logged_in_api, model_registry, object_store
):
    files = {"config.json": QUANTIZED_CONFIG}
    register(model_registry, object_store, "qwen-fp8", 1, files=files, weights="full")
    request = pipeline_request()
    del request["finetune"]["base_model"]
    request["finetune"]["from"] = "model:qwen-fp8"

    [error] = errors(logged_in_api, request)

    assert error["loc"] == ["finetune", "from"]
    assert "quantized" in error["msg"]


def test_an_adapter_is_quantized_with_its_base_model_fetched_for_the_merge(
    logged_in_api, cluster, model_registry, object_store
):
    register_adapter(model_registry, object_store, "qwen-sft", PINNED_BASE_MODEL)

    response = submit(logged_in_api, quantize_only("model:qwen-sft"))

    assert response.status_code == 202, response.text
    assert set(tasks(cluster)) == {"fetch", "quantize"}
    assert tasks(cluster)["quantize"]["dependentTasks"] == ["fetch"]
    assert parameters(tasks(cluster)["fetch"])["references"] == PINNED_BASE_MODEL
    request = json.loads(parameters(tasks(cluster)["quantize"])["request"])
    assert request["quantize"]["model"] == "model:qwen-sft@1"


def test_quantize_is_handed_the_model_version_of_the_last_phase(logged_in_api, cluster):
    pipeline_id = submit(logged_in_api, quantizing()).json()["id"]

    quantize = tasks(cluster)["quantize"]
    assert quantize["dependentTasks"] == ["finetune"]
    assert parameters(quantize)["pipeline_id"] == str(pipeline_id)
    assert parameters(quantize)["finetuned"] == "finetune"


def test_quantize_loads_the_model_offline_on_the_platforms_gpus(logged_in_api, cluster):
    submit(logged_in_api, quantizing())

    step = container(cluster, "quantize")
    assert step["image"] == "mlp-stages:test"
    assert step["command"] == ["mlp-stage", "quantize"]
    assert step["resources"]["accelerator"]["resourceCount"] == "1"
    assert {"name": "HF_HUB_OFFLINE", "value": "1"} in step["env"]
    assert {"name": "HF_HOME", "value": "/model-cache"} in step["env"]
    assert {"name": "S3_BUCKET", "value": "platform"} in step["env"]


def test_evaluate_takes_the_quantized_model(logged_in_api, cluster):
    request = {**quantizing(), "evaluate": {"benchmarks": [GSM8K]}}

    submit(logged_in_api, request)

    assert tasks(cluster)["evaluate"]["dependentTasks"] == ["quantize"]
    evaluated = json.loads(parameters(tasks(cluster)["evaluate"])["request"])
    assert evaluated["evaluate"]["model"] == "@quantize"


@pytest.mark.parametrize(
    ("request_", "loc", "message"),
    [
        (quantize_only("@finetune"), ["quantize", "model"], "`finetune` isn't enabled"),
        (
            {"name": "e", "evaluate": {"model": "@quantize", "benchmarks": [GSM8K]}},
            ["evaluate", "model"],
            "`quantize` isn't enabled",
        ),
    ],
)
def test_a_handoff_to_a_stage_that_is_not_enabled_is_rejected(
    logged_in_api, request_, loc, message
):
    [error] = errors(logged_in_api, request_)

    assert error["loc"] == loc
    assert message in error["msg"]


def test_quantize_is_listed_as_a_stage(logged_in_api):
    submit(logged_in_api, quantizing())

    [pipeline] = logged_in_api.get(api_paths.PIPELINES).json()
    assert pipeline["stages"] == ["finetune", "quantize"]


def test_a_resume_quantizes_the_model_versions_its_phases_registered_before(
    logged_in_api, cluster, model_registry
):
    failed = failed_pipeline(logged_in_api, quantizing())
    registered_phase(logged_in_api, failed, 0, 1)
    # The quantized copy the failed Pipeline made, which has no Phase.
    registered_phase(logged_in_api, failed, 0, 2)
    del model_registry.versions[-1].tags["phase"]
    model_registry.versions[-1].tags["quantization"] = "fp8-dynamic"

    response = resume(logged_in_api, failed)

    assert response.status_code == 202, response.text
    assert set(tasks(cluster)) == {"fetch", "quantize"}
    assert parameters(tasks(cluster)["quantize"])["finetuned"] == "model:qwen-sft@1"


def test_the_install_registers_the_default_calibration_dataset_once(
    logged_in_api, hugging_face, object_store
):
    source = config.CALIBRATION_SOURCE
    rows = jsonl({"text": "Hi", "messages": [{"role": "user", "content": "Hi"}]})
    hugging_face.dataset_files[source["repo"]] = {source["file"]: gzip.compress(rows)}
    state = logged_in_api.app.state

    for _ in range(2):
        register_calibration_dataset(state.engine, object_store, hugging_face)

    [version] = datasets(logged_in_api)[config.CALIBRATION_DATASET]
    assert (version["version"], version["row_format"]) == (1, "messages")
    assert object_store.objects[f"datasets/{config.CALIBRATION_DATASET}/1/data.jsonl"] == rows
