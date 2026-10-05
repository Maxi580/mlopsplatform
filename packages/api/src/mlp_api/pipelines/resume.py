from mlp_api.datasets.pipeline_output import find_distillation_dataset
from mlp_api.pipelines.compiler import Resume
from mlp_api.pipelines.lifecycle import find_pipeline
from mlp_core import config
from mlp_core.pipeline_request.references import checkpoint_prefix, model_reference
from mlp_core.pipeline_request.schema import PipelineRequest


def plan_resume(state, pipeline_id: int) -> Resume:
    """What a Pipeline resuming this one reuses; LookupError if unknown, ValueError if it can't."""
    # 1. A failed or cancelled Pipeline, and what it reused itself if it was a resume too.
    row = find_pipeline(state.engine, pipeline_id)
    if row.cases is not None:
        raise ValueError(f"Smoke Test {row.name} can't be resumed; start a new one")
    if row.status not in config.RESUMABLE_STATUSES:
        raise ValueError(f"Pipeline {pipeline_id} is {row.status}; only failed or cancelled resume")
    earlier = Resume(**(row.resume or {}))
    request = PipelineRequest.model_validate(row.request)

    # 2. The Model Versions of the Phases finished so far, by this Pipeline or the ones before it,
    # unless deleted since.
    registered = state.model_registry.model_versions()
    existing = {model_reference(found.name, found.version) for found in registered}
    finished = {i: ref for i, ref in enumerate(earlier.model_versions) if ref in existing}
    for found in registered:
        made_here = found.name == request.name and found.tags.get("pipeline") == str(pipeline_id)
        # A quantized copy has no Phase; `quantize` runs again.
        if made_here and "phase" in found.tags:
            finished[int(found.tags["phase"]) - 1] = model_reference(found.name, found.version)
    model_versions = []
    while len(model_versions) in finished:
        model_versions.append(finished[len(model_versions)])

    # 3. The next Phase's Checkpoint: its own, else the one it was handed and saved none after.
    index, store = len(model_versions), state.object_store
    candidates = [checkpoint_prefix(pipeline_id, index)]
    if earlier.checkpoint and index == len(earlier.model_versions):
        candidates.append(earlier.checkpoint)
    checkpoint = next((c for c in candidates if any(store.list_objects(store.bucket, c))), None)

    # 4. The Distillation Dataset its `distill` Stage registered.
    distilled = earlier.distilled_dataset or find_distillation_dataset(state.engine, pipeline_id)
    return Resume(model_versions, distilled, checkpoint)
