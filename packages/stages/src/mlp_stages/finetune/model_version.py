from mlp_core.pipeline_request.schema import PipelineRequest
from mlp_stages.model_versions import renders_tools, tool_parser_of


def phase_tags(
    request: PipelineRequest,
    pipeline_id: str,
    phase_index: int,
    parent: str,
    base: str,
    tokenizer,
    model_type: str,
) -> dict[str, str]:
    """The lineage tags of the Model Version the Phase registers."""
    finetune = request.finetune
    phase = finetune.phases[phase_index]
    return {
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
        # The Base Model or Model Version a `distillation` Phase learned from.
        **({"teacher": phase.teacher} if phase.teacher else {}),
        "method": phase.method,
        "backend": finetune.backend,
        "tool_parser": tool_parser_of(finetune.starting_model, model_type),
        "tools_rendered": str(renders_tools(tokenizer)).lower(),
    }
