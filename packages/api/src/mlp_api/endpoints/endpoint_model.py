import json
from dataclasses import dataclass, field, replace

from mlp_api.models.mlflow import MLflow
from mlp_api.models.registry import find_referenced_model_version
from mlp_api.pipelines.hugging_face import HuggingFace, find_base_model
from mlp_api.storage.object_store import ObjectStore
from mlp_core import config
from mlp_core.endpoint_spec import VllmModel
from mlp_core.pipeline_request.references import model_reference, split_base_model_reference


@dataclass(frozen=True)
class EndpointModel:
    """What vLLM loads, and what the Endpoint's pod fetches or downloads for it first."""

    vllm: VllmModel
    # Every Reference the Endpoint uses, pinned, its model first.
    references: list[str]
    # The Base Model fetch puts into the Model Cache, if vLLM loads one.
    base_model: str | None = None
    # `s3://<bucket>/<prefix>` of Model Version files -> the directory they are downloaded to.
    downloads: dict[str, str] = field(default_factory=dict)


def find_endpoint_model(
    reference: str, hugging_face: HuggingFace, model_registry: MLflow, object_store: ObjectStore
) -> EndpointModel:
    """What vLLM loads for the `hf:` or `model:` Reference; ValueError saying why it can't."""
    # 1. A Base Model, from the Model Cache; Endpoints have no Hugging Face token.
    if reference.startswith("hf:"):
        pinned, model = find_base_model(hugging_face, reference, None)
        repo, commit = split_base_model_reference(pinned)
        tool_parser = config.TOOL_PARSERS.get(model.model_type)
        return EndpointModel(VllmModel(repo, commit, tool_parser=tool_parser), [pinned], pinned)

    # 2. Full weights, from the Model Version's files.
    found = find_referenced_model_version(model_registry, reference)
    pinned = model_reference(found.name, found.version)
    bucket = model_registry.artifact_bucket
    files = f"s3://{bucket}/{found.artifact_prefix}"
    if found.tags.get("weights") != "adapter":
        vllm = VllmModel(config.ENDPOINT_WEIGHTS_DIRECTORY, tool_parser=tool_parser_tag(found.tags))
        return EndpointModel(vllm, [pinned], downloads={files: config.ENDPOINT_WEIGHTS_DIRECTORY})

    # 3. An Adapter on its base; an uploaded one has no parser of its own and takes its base's.
    base = find_endpoint_model(found.tags["base_model"], hugging_face, model_registry, object_store)
    adapter_config = object_store.read(bucket, f"{found.artifact_prefix}adapter_config.json")
    vllm = replace(
        base.vllm,
        adapter_path=config.ENDPOINT_ADAPTER_DIRECTORY,
        adapter_rank=json.loads(adapter_config)["r"],
        tool_parser=tool_parser_tag(found.tags, base.vllm.tool_parser),
    )
    downloads = {**base.downloads, files: config.ENDPOINT_ADAPTER_DIRECTORY}
    return EndpointModel(vllm, [pinned, *base.references], base.base_model, downloads)


# Model Versions tag `none` for a model served without tool calling.
def tool_parser_tag(tags: dict[str, str], default: str | None = None) -> str | None:
    tool_parser = tags.get("tool_parser", default)
    return None if tool_parser == "none" else tool_parser
