from pathlib import Path

import mlflow
from huggingface_hub import snapshot_download
from mlflow.exceptions import MlflowException

from mlp_core import config
from mlp_core.pipeline_request.references import (
    model_reference,
    split_base_model_reference,
    split_model_reference,
)
from mlp_core.pipeline_request.schema import PipelineRequest


def model_with_adapter(reference: str, scratch: Path) -> tuple[str, Path, Path | None]:
    """The model's base, the base's files, and the files of the model's Adapter, if it is one."""
    files = model_files(reference, scratch / "model")
    if reference.startswith("hf:"):
        return reference, files, None
    tags = model_version_tags(reference)
    if tags["weights"] != "adapter":
        return reference, files, None
    base = tags["base_model"]
    return base, model_files(base, scratch / "base"), files


def model_files(reference: str, destination: Path) -> Path:
    """Where the `hf:` or `model:` Reference's files lie: the Model Cache, or a download."""
    if reference.startswith("hf:"):
        repo, commit = split_base_model_reference(reference)
        # fetch put it in the Model Cache; offline, this only finds it there.
        return Path(snapshot_download(repo, revision=commit))
    name, version = split_model_reference(reference)
    uri = f"models:/{name}/{version}"
    return Path(mlflow.artifacts.download_artifacts(uri, dst_path=str(destination)))


def model_version_tags(reference: str) -> dict[str, str]:
    name, version = split_model_reference(reference)
    return mlflow.MlflowClient().get_model_version(name, str(version)).tags


# A Model Version keeps its parser, which an upload may name for an unknown model_type; an Adapter
# without one takes its base's.
def tool_parser_of(reference: str, model_type: str) -> str:
    """The tool parser of a model built from the Reference."""
    if reference.startswith("model:"):
        tags = model_version_tags(reference)
        if "tool_parser" in tags:
            return tags["tool_parser"]
        if tags["weights"] == "adapter":
            return tool_parser_of(tags["base_model"], model_type)
    return config.TOOL_PARSERS.get(model_type, "none")


def register_model_version(
    model_directory: Path, run_id: str, request: PipelineRequest, tags: dict[str, str]
) -> str:
    """The `model:` Reference of the files, logged into the Run as the Pipeline's next version."""
    # 1. The resolved request travels with the weights.
    (model_directory / "pipeline_request.json").write_text(request.model_dump_json(indent=1))

    # 2. Into the step's Run, where the step logged its params and metrics.
    client = mlflow.MlflowClient()
    client.log_artifacts(run_id, str(model_directory), "model")

    # 3. The next Model Version, with its lineage; another Pipeline may create the name first.
    try:
        client.create_registered_model(request.name)
    except MlflowException as error:
        if error.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
    registered = client.create_model_version(
        request.name,
        source=f"{client.get_run(run_id).info.artifact_uri}/model",
        run_id=run_id,
        tags=tags,
    )
    return model_reference(request.name, int(registered.version))


# A template that drops the tools a client sends can't teach or serve tool calls (#16).
def renders_tools(tokenizer) -> bool:
    if tokenizer.chat_template is None:
        return False
    tool = {
        "type": "function",
        "function": {
            "name": "probe_tool",
            "description": "Shows whether the template renders tools.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    conversation = [{"role": "user", "content": "Hi"}]
    try:
        text = tokenizer.apply_chat_template(conversation, tools=[tool], tokenize=False)
    except Exception:
        return False
    return "probe_tool" in text
