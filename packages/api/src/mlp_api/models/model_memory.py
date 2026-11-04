import json
from dataclasses import replace

from mlp_api.models.mlflow import MLflow
from mlp_api.models.registry import find_referenced_model_version
from mlp_api.pipelines.hugging_face import HuggingFace
from mlp_api.storage.object_store import ObjectStore
from mlp_core import config
from mlp_core.gpu_share import ModelMemory, model_memory
from mlp_core.pipeline_request.references import split_base_model_reference


def find_model_memory(
    model: str,
    token: str | None,
    hugging_face: HuggingFace,
    model_registry: MLflow,
    object_store: ObjectStore,
) -> ModelMemory | None:
    """What the pinned model takes on a GPU, from its files; None if they don't say."""
    # 1. Only a Base Model or Model Version has files; a running Endpoint holds its own share.
    if not model.startswith(("hf:", "model:")):
        return None

    # 2. An Adapter is its base plus its own file.
    if model.startswith("model:"):
        found = find_referenced_model_version(model_registry, model)
        if found.tags.get("weights") == "adapter":
            base = find_model_memory(
                found.tags["base_model"], token, hugging_face, model_registry, object_store
            )
            adapter = found.artifact_prefix + config.ADAPTER_WEIGHTS_FILE
            size = object_store.size_of(model_registry.artifact_bucket, adapter)
            return base and replace(base, weights_bytes=base.weights_bytes + size)

    # 3. The config.json and the size of the weights.
    read = model_file_reader(model, token, hugging_face, model_registry, object_store)
    model_config = read("config.json")
    weights_bytes = weights_size(model, read, token, hugging_face, model_registry, object_store)
    if model_config is None or weights_bytes is None:
        return None
    try:
        return model_memory(json.loads(model_config), weights_bytes)
    except ValueError:
        return None


def model_file_reader(
    model: str,
    token: str | None,
    hugging_face: HuggingFace,
    model_registry: MLflow,
    object_store: ObjectStore,
):
    """Reads one file of the pinned Base Model or Model Version, or None if it has none."""
    if model.startswith("hf:"):
        repo, commit = split_base_model_reference(model)
        return lambda path: hugging_face.model_file(repo, commit, path, token)
    found = find_referenced_model_version(model_registry, model)
    bucket = model_registry.artifact_bucket
    keys = {item["Key"] for item in object_store.list_objects(bucket, found.artifact_prefix)}
    return lambda path: (
        object_store.read(bucket, found.artifact_prefix + path)
        if found.artifact_prefix + path in keys
        else None
    )


def weights_size(
    model: str,
    read,
    token: str | None,
    hugging_face: HuggingFace,
    model_registry: MLflow,
    object_store: ObjectStore,
) -> int | None:
    """The bytes of the weights alone: all of a repo's files can hold them twice over."""
    # 1. The index of sharded weights.
    index = read(config.WEIGHTS_INDEX_FILE)
    if index is not None:
        try:
            return json.loads(index)["metadata"]["total_size"]
        except (ValueError, KeyError, TypeError):
            return None

    # 2. Else the one file they fit in.
    if model.startswith("hf:"):
        repo, commit = split_base_model_reference(model)
        return hugging_face.model_file_size(repo, commit, config.WEIGHTS_FILE, token)
    found = find_referenced_model_version(model_registry, model)
    file = found.artifact_prefix + config.WEIGHTS_FILE
    return object_store.size_of(model_registry.artifact_bucket, file) or None
