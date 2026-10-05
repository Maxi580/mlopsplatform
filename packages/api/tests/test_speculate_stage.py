import json

import pytest

from mlp_core import api_paths

from .test_datasets import jsonl, upload
from .test_distill_stage import distilling, prompts  # noqa: F401
from .test_endpoints import register, register_adapter
from .test_evaluate_stage import GSM8K, PINNED_BASE_MODEL, container, errors, tasks, validate
from .test_pipeline_request import BASE_MODEL, pipeline_request
from .test_pipelines import submit, submittable  # noqa: F401
from .test_quantize_stage import quantizing, resolved
from .test_resume import parameters

# Every test submits or validates a request for the Base Model and the `chat` Dataset.
pytestmark = pytest.mark.usefixtures("submittable")


def merged_finetune() -> dict:
    """The finetune request, its Phase registering full weights a Speculator can read."""
    return pipeline_request(phase={"output": "merged"})


def speculating(request=None, **speculate) -> dict:
    request = request or merged_finetune()
    return {**request, "speculate": {"dataset": "dataset:chat", **speculate}}


def speculate_only(model=f"hf:{BASE_MODEL}", **speculate) -> dict:
    return speculating({"name": "qwen"}, model=model, **speculate)


def test_speculate_defaults_to_the_output_of_finetune_with_the_default_settings(logged_in_api):
    speculate = resolved(logged_in_api, speculating())["speculate"]

    assert speculate == {
        "speculator": "eagle3",
        "model": "@finetune",
        "dataset": "dataset:chat@1",
        "settings": {
            "samples": 1000,
            "seq_length": 8192,
            "epochs": 5,
            "learning_rate": 1e-4,
            "draft_vocab_size": 32000,
        },
    }


def test_speculate_defaults_to_the_quantized_model_when_quantize_runs(logged_in_api):
    request = speculating(quantizing())

    assert resolved(logged_in_api, request)["speculate"]["model"] == "@quantize"


def test_a_base_model_gets_a_speculator_without_finetune_pinned_to_a_commit(logged_in_api):
    speculate = resolved(logged_in_api, speculate_only(speculator="dflash"))["speculate"]

    assert (speculate["model"], speculate["speculator"]) == (PINNED_BASE_MODEL, "dflash")


def test_speculate_without_a_model_needs_one_named(logged_in_api):
    [error] = errors(logged_in_api, speculating({"name": "qwen"}))

    assert error["loc"] == ["speculate", "model"]
    assert "name the verifier" in error["msg"]


def test_an_adapter_from_finetune_is_no_verifier(logged_in_api):
    [error] = errors(logged_in_api, speculating(pipeline_request()))

    assert error["loc"] == ["speculate", "model"]
    assert "set `output: merged`" in error["msg"]


def test_an_adapter_model_version_is_no_verifier(logged_in_api, model_registry, object_store):
    register_adapter(model_registry, object_store, "qwen-sft", PINNED_BASE_MODEL)

    [error] = errors(logged_in_api, speculate_only("model:qwen-sft"))

    assert "model:qwen-sft@1 is an Adapter" in error["msg"]


def test_a_speculator_is_no_verifier(logged_in_api, model_registry, object_store):
    tags = {"weights": "speculator", "speculator": "eagle3", "verifier": PINNED_BASE_MODEL}
    register(model_registry, object_store, "qwen-speculator", tags)

    [error] = errors(logged_in_api, speculate_only("model:qwen-speculator"))

    assert "is a Speculator" in error["msg"]


def test_speculate_trains_on_messages_rows(logged_in_api):
    upload(logged_in_api, "essays", jsonl({"text": "Once upon a time"}))

    [error] = errors(logged_in_api, speculate_only(dataset="dataset:essays"))

    assert error["loc"] == ["speculate", "dataset"]
    assert "`speculate` trains on messages rows" in error["msg"]


@pytest.mark.usefixtures("prompts")
def test_speculate_can_train_on_the_distilled_replies(logged_in_api):
    request = {**distilling(), **speculate_only(dataset="@distill"), "name": "qwen-distill"}

    assert validate(logged_in_api, request).status_code == 200


def test_distilled_replies_need_distill_enabled(logged_in_api):
    [error] = errors(logged_in_api, speculate_only(dataset="@distill"))

    assert error["loc"] == ["speculate", "dataset"]
    assert "`distill` isn't enabled" in error["msg"]


@pytest.mark.parametrize("settings", [{"samples": 0}, {"learning_rate": -1}, {"workers": 4}])
def test_invalid_or_unknown_settings_are_rejected(logged_in_api, settings):
    assert validate(logged_in_api, speculate_only(settings=settings)).status_code == 422


def test_speculate_is_handed_the_outputs_of_the_stages_before_it(logged_in_api, cluster):
    # The quantized model is full weights, so the Phase may keep an Adapter.
    pipeline_id = submit(logged_in_api, speculating(quantizing())).json()["id"]

    speculate = tasks(cluster)["speculate"]
    assert speculate["dependentTasks"] == ["finetune", "quantize"]
    assert parameters(speculate) == {
        "pipeline_id": str(pipeline_id),
        "request": parameters(speculate)["request"],
        "finetuned": "finetune",
        "quantized": "quantize",
        "distilled_dataset": "",
        "gpus": "1",
        "dataloader_workers": "4",
    }


def test_speculate_trains_offline_on_the_platforms_gpus(logged_in_api, cluster):
    submit(logged_in_api, speculating())

    step = container(cluster, "speculate")
    assert step["image"] == "mlp-stages:test"
    assert step["command"] == ["mlp-stage", "speculate"]
    assert step["resources"]["accelerator"]["resourceCount"] == "1"
    assert {"name": "HF_HUB_OFFLINE", "value": "1"} in step["env"]
    assert {"name": "S3_BUCKET", "value": "platform"} in step["env"]


def test_the_verifiers_base_model_is_fetched(logged_in_api, cluster):
    submit(logged_in_api, speculate_only())

    assert set(tasks(cluster)) == {"fetch", "speculate"}
    assert parameters(tasks(cluster)["fetch"])["references"] == PINNED_BASE_MODEL
    request = json.loads(parameters(tasks(cluster)["speculate"])["request"])
    assert request["speculate"]["model"] == PINNED_BASE_MODEL


def test_speculate_runs_before_evaluate(logged_in_api, cluster):
    submit(logged_in_api, {**speculating(), "evaluate": {"benchmarks": [GSM8K]}})

    assert tasks(cluster)["speculate"]["dependentTasks"] == ["finetune"]
    assert tasks(cluster)["evaluate"]["dependentTasks"] == ["speculate"]


def test_speculate_is_listed_as_a_stage(logged_in_api):
    submit(logged_in_api, speculating())

    [pipeline] = logged_in_api.get(api_paths.PIPELINES).json()
    assert pipeline["stages"] == ["finetune", "speculate"]
