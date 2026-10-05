from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException

from mlp_core import config
from mlp_core.pipeline_request.references import model_reference, split_model_reference
from mlp_core.pipeline_request.schema import PipelineRequest


def register_model_version(
    model_directory: Path,
    run_id: str,
    request: PipelineRequest,
    pipeline_id: str,
    phase_index: int,
    parent: str,
    base: str,
    tokenizer,
    model_type: str,
) -> str:
    """The `model:` Reference of the Phase's output, logged into the step's Run and registered."""
    # 1. The resolved request travels with the weights.
    (model_directory / "pipeline_request.json").write_text(request.model_dump_json(indent=1))

    # 2. Into the step's Run, where the trainer logged its params and metrics.
    client = mlflow.MlflowClient()
    client.log_artifacts(run_id, str(model_directory), "model")

    # 3. The next Model Version, with its lineage; another Pipeline may create the name first.
    try:
        client.create_registered_model(request.name)
    except MlflowException as error:
        if error.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
    finetune = request.finetune
    phase = finetune.phases[phase_index]
    tool_parser = config.TOOL_PARSERS.get(model_type, "none")
    if finetune.from_:
        # A Model Version keeps its parser, which an upload may name for an unknown model_type.
        name, version = split_model_reference(finetune.from_)
        starting = client.get_model_version(name, str(version))
        tool_parser = starting.tags.get("tool_parser", tool_parser)
    registered = client.create_model_version(
        request.name,
        source=f"{client.get_run(run_id).info.artifact_uri}/model",
        run_id=run_id,
        tags={
            # The Adapter's base: a Base Model or a full-weight Model Version.
            **(
                {"weights": "adapter", "base_model": base}
                if phase.keeps_adapter
                else {"weights": "full"}
            ),
            # An Adapter merged into its base, registered in its place.
            **({"merged": "true"} if phase.merges_adapter else {}),
            # What the Phase started from: the previous Phase's Model Version, or the base.
            "parent": parent,
            "pipeline": pipeline_id,
            "phase": str(phase_index + 1),
            "algorithm": phase.algorithm,
            "method": phase.method,
            "backend": finetune.backend,
            "tool_parser": tool_parser,
            "tools_rendered": str(renders_tools(tokenizer)).lower(),
        },
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
