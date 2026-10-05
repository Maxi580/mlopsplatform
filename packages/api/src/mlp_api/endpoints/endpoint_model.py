import json
from dataclasses import dataclass, field, replace

from mlp_api.models.mlflow import MLflow
from mlp_api.models.registry import find_referenced_model_version
from mlp_api.pipelines.hugging_face import HuggingFace, find_base_model
from mlp_api.storage.object_store import ObjectStore
from mlp_core import config
from mlp_core.endpoint_spec import Speculative, VllmModel
from mlp_core.pipeline_request.references import model_reference, split_base_model_reference


@dataclass(frozen=True)
class EndpointModel:
    """What vLLM loads, and what the Endpoint's pod fetches or downloads for it first."""

    vllm: VllmModel
    # Every Reference the Endpoint uses, pinned, its model first.
    references: list[str]
    # The Base Models fetch puts into the Model Cache, for vLLM to load.
    base_models: list[str] = field(default_factory=list)
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
        return EndpointModel(VllmModel(repo, commit, tool_parser=tool_parser), [pinned], [pinned])

    # 2. Full weights, from the Model Version's files.
    found = find_referenced_model_version(model_registry, reference)
    pinned = model_reference(found.name, found.version)
    if "speculator" in found.tags:
        verifier = found.tags["verifier"]
        raise ValueError(f"{pinned} is a Speculator; serve {verifier} with it in `speculative`")
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
    return EndpointModel(vllm, [pinned, *base.references], base.base_models, downloads)


def add_drafter(
    model: EndpointModel,
    speculative: Speculative,
    hugging_face: HuggingFace,
    model_registry: MLflow,
) -> tuple[EndpointModel, str | None]:
    """The model with the drafter vLLM loads beside it, and the drafter pinned; else ValueError."""
    # 1. `ngram` drafts from the context.
    if speculative.model is None:
        return model, None

    # 2. A Base Model, from the Model Cache; a Speculator on the hub is the hub's to answer for.
    if speculative.model.startswith("hf:"):
        pinned, _ = find_base_model(hugging_face, speculative.model, None)
        repo, commit = split_base_model_reference(pinned)
        vllm = replace(model.vllm, drafter_path=repo, drafter_revision=commit)
        references, base_models = [*model.references, pinned], [*model.base_models, pinned]
        return replace(model, vllm=vllm, references=references, base_models=base_models), pinned

    # 3. A Model Version: a Speculator trained for exactly this model, or full weights for `draft`.
    found = find_referenced_model_version(model_registry, speculative.model)
    pinned, served = model_reference(found.name, found.version), model.references[0]
    speculator = found.tags.get("speculator")
    if speculator:
        if found.tags["verifier"] != served:
            raise ValueError(f"{pinned} was trained for {found.tags['verifier']}, not {served}")
        if speculative.method != speculator:
            msg = f"{pinned} is a {speculator} Speculator; serve it with method `{speculator}`"
            raise ValueError(msg)
    elif speculative.method != "draft":
        raise ValueError(f"{pinned} is no Speculator; draft with it as method `draft`")
    elif found.tags.get("weights") == "adapter":
        raise ValueError(f"{pinned} is an Adapter; `draft` drafts with full weights")
    files = f"s3://{model_registry.artifact_bucket}/{found.artifact_prefix}"
    vllm = replace(model.vllm, drafter_path=config.ENDPOINT_DRAFTER_DIRECTORY)
    downloads = {**model.downloads, files: config.ENDPOINT_DRAFTER_DIRECTORY}
    references = [*model.references, pinned]
    return replace(model, vllm=vllm, references=references, downloads=downloads), pinned


# Model Versions tag `none` for a model served without tool calling.
def tool_parser_tag(tags: dict[str, str], default: str | None = None) -> str | None:
    tool_parser = tags.get("tool_parser", default)
    return None if tool_parser == "none" else tool_parser
