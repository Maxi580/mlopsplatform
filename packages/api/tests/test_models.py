from mlp_api.models.mlflow import ModelVersion
from mlp_core import api_paths


def register(
    model_registry, object_store, name, version, pipeline_id=1, files=None, weights="adapter"
):
    """A Model Version whose files sit in the mlflow bucket, as the finetune Stage leaves them."""
    prefix = f"1/run-{name}-{version}/artifacts/model/"
    for file, content in (files or {"adapter_model.safetensors": b"weights"}).items():
        object_store.buckets["mlflow"][prefix + file] = content
    tags = {"weights": weights, "base_model": "hf:Qwen/Qwen3@abc", "pipeline": str(pipeline_id)}
    model_registry.versions.append(ModelVersion(name, version, tags, prefix))


def models(api) -> dict:
    response = api.get(api_paths.MODELS)
    assert response.status_code == 200, response.text
    return {model["name"]: model["versions"] for model in response.json()}


def delete(api, name, version):
    return api.delete(api_paths.MODEL_VERSION.format(name=name, version=version))


def test_the_list_shows_versions_with_sizes_and_lineage(
    logged_in_api, model_registry, object_store
):
    files = {"adapter_model.safetensors": b"x" * 100, "adapter_config.json": b"{}"}
    register(model_registry, object_store, "qwen-sft", 2, files=files)
    register(model_registry, object_store, "qwen-sft", 1)

    [first, second] = models(logged_in_api)["qwen-sft"]

    assert (first["version"], first["size_bytes"]) == (1, len(b"weights"))
    assert (second["version"], second["size_bytes"]) == (2, 102)
    assert second["tags"]["weights"] == "adapter"
    assert second["tags"]["pipeline"] == "1"


def test_deleting_an_unused_version_removes_it_and_its_files(
    logged_in_api, model_registry, object_store
):
    register(model_registry, object_store, "qwen-sft", 1)
    register(model_registry, object_store, "qwen-sft", 2)

    response = delete(logged_in_api, "qwen-sft", 1)

    assert response.status_code == 204, response.text
    assert [v["version"] for v in models(logged_in_api)["qwen-sft"]] == [2]
    assert list(object_store.buckets["mlflow"]) == [
        "1/run-qwen-sft-2/artifacts/model/adapter_model.safetensors"
    ]


def test_deleting_an_unknown_version_is_not_found(logged_in_api, model_registry, object_store):
    register(model_registry, object_store, "qwen-sft", 1)

    assert delete(logged_in_api, "qwen-sft", 2).status_code == 404


def test_models_require_login(api, model_registry):
    assert api.get(api_paths.MODELS).status_code == 401
    assert delete(api, "qwen-sft", 1).status_code == 401
