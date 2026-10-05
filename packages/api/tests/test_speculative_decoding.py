import json

import pytest

from .test_endpoints import (
    COMMIT,
    PINNED_QWEN,
    QWEN,
    init_commands,
    qwen_on_the_hub,  # noqa: F401
    register,
    register_adapter,
    start,
    vllm_args,
)

pytestmark = pytest.mark.usefixtures("qwen_on_the_hub")


def register_speculator(model_registry, object_store, verifier=PINNED_QWEN, speculator="eagle3"):
    """Version 1 of `qwen-speculator`, a Speculator trained for the verifier; its file prefix."""
    tags = {"weights": "speculator", "speculator": speculator, "verifier": verifier}
    files = {"config.json": b"{}", "model.safetensors": b"drafter"}
    return register(model_registry, object_store, "qwen-speculator", tags, files)


def speculative_config(cluster) -> dict:
    args = vllm_args(cluster)
    return json.loads(args[args.index("--speculative-config") + 1])


def test_a_speculator_for_the_served_model_drafts_from_its_downloaded_files(
    logged_in_api, cluster, model_registry, object_store
):
    prefix = register_speculator(model_registry, object_store)
    speculative = {"method": "eagle3", "model": "model:qwen-speculator"}

    response = start(logged_in_api, f"hf:{QWEN}", speculative=speculative)

    assert response.status_code == 201, response.text
    assert response.json()["spec"]["speculative"]["model"] == "model:qwen-speculator@1"
    assert init_commands(cluster) == [
        ["mlp-stage", "fetch", "200Gi", PINNED_QWEN],
        ["mlp-stage", "download", f"s3://mlflow/{prefix}=/models/drafter"],
    ]
    assert speculative_config(cluster) == {
        "method": "eagle3",
        "model": "/models/drafter",
        "num_speculative_tokens": 3,
    }


def test_a_speculator_for_another_model_is_rejected(
    logged_in_api, cluster, model_registry, object_store
):
    register_speculator(model_registry, object_store, verifier="hf:org/other@" + "b" * 40)
    speculative = {"method": "eagle3", "model": "model:qwen-speculator@1"}

    response = start(logged_in_api, speculative=speculative)

    assert response.status_code == 422
    assert "was trained for hf:org/other" in response.json()["detail"]
    assert cluster.endpoints == {}


def test_a_speculator_for_an_adapters_base_does_not_fit_the_adapter(
    logged_in_api, cluster, model_registry, object_store
):
    register_adapter(model_registry, object_store, "qwen-sft", PINNED_QWEN)
    register_speculator(model_registry, object_store)
    speculative = {"method": "eagle3", "model": "model:qwen-speculator@1"}

    response = start(logged_in_api, "model:qwen-sft@1", speculative=speculative)

    assert response.status_code == 422
    assert "not model:qwen-sft@1" in response.json()["detail"]


def test_a_speculator_serves_with_its_own_method(
    logged_in_api, cluster, model_registry, object_store
):
    register_speculator(model_registry, object_store, speculator="dflash")
    speculative = {"method": "eagle3", "model": "model:qwen-speculator@1"}

    response = start(logged_in_api, speculative=speculative)

    assert response.status_code == 422
    assert "a dflash Speculator; serve it with method `dflash`" in response.json()["detail"]


def test_a_peagle_speculator_drafts_its_tokens_in_parallel(
    logged_in_api, cluster, model_registry, object_store
):
    register_speculator(model_registry, object_store, speculator="peagle")
    speculative = {
        "method": "peagle",
        "model": "model:qwen-speculator",
        "num_speculative_tokens": 5,
    }

    response = start(logged_in_api, speculative=speculative)

    assert response.status_code == 201, response.text
    assert speculative_config(cluster) == {
        "method": "eagle3",
        "parallel_drafting": True,
        "model": "/models/drafter",
        "num_speculative_tokens": 5,
    }


def test_a_model_version_that_is_no_speculator_cannot_draft_as_one(
    logged_in_api, cluster, model_registry, object_store
):
    register(model_registry, object_store, "my-model", {"weights": "full"})
    speculative = {"method": "eagle3", "model": "model:my-model"}

    response = start(logged_in_api, speculative=speculative)

    assert response.status_code == 422
    assert "model:my-model@1 is no Speculator" in response.json()["detail"]


def test_a_speculator_is_not_served_as_the_model(logged_in_api, model_registry, object_store):
    register_speculator(model_registry, object_store)

    response = start(logged_in_api, "model:qwen-speculator")

    assert response.status_code == 422
    assert "is a Speculator" in response.json()["detail"]


def test_ngram_drafts_from_the_context_without_a_model(logged_in_api, cluster):
    speculative = {"method": "ngram", "prompt_lookup_min": 2, "prompt_lookup_max": 4}

    response = start(logged_in_api, speculative=speculative)

    assert response.status_code == 201, response.text
    assert speculative_config(cluster) == {
        "method": "ngram",
        "num_speculative_tokens": 3,
        "prompt_lookup_min": 2,
        "prompt_lookup_max": 4,
    }
    assert init_commands(cluster) == [["mlp-stage", "fetch", "200Gi", PINNED_QWEN]]


def test_a_small_base_model_drafts_from_the_model_cache(logged_in_api, hugging_face, cluster):
    hugging_face.models["Qwen/Qwen3-0.6B-draft"] = hugging_face.models[QWEN]
    speculative = {"method": "draft", "model": "hf:Qwen/Qwen3-0.6B-draft"}

    response = start(logged_in_api, speculative=speculative)

    assert response.status_code == 201, response.text
    drafter = f"hf:Qwen/Qwen3-0.6B-draft@{COMMIT}"
    assert init_commands(cluster) == [["mlp-stage", "fetch", "200Gi", f"{PINNED_QWEN},{drafter}"]]
    assert speculative_config(cluster) == {
        "method": "draft_model",
        "model": "Qwen/Qwen3-0.6B-draft",
        "revision": COMMIT,
        "num_speculative_tokens": 3,
    }


@pytest.mark.parametrize(
    "speculative",
    [
        {"method": "ngram", "model": "hf:Qwen/Qwen3-0.6B"},
        {"method": "eagle3"},
        {"method": "eagle3", "model": "model:qwen-speculator", "prompt_lookup_max": 4},
        {"method": "medusa", "model": "model:qwen-speculator"},
    ],
)
def test_a_speculative_block_that_does_not_fit_its_method_is_rejected(
    logged_in_api, cluster, speculative
):
    response = start(logged_in_api, speculative=speculative)

    assert response.status_code == 422
    assert cluster.endpoints == {}
