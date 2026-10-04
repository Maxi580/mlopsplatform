from pathlib import Path

from mlp_api.datasets.registry import upload_dataset_version
from mlp_api.pipelines.lifecycle import find_running_request
from mlp_core import config
from mlp_core.pipeline_request.references import dataset_reference


def register_distillation_dataset(state, pipeline_id: int, path: Path) -> str:
    """The `dataset:` Reference of the Pipeline's `distill` output, registered under its name."""
    request = find_running_request(state.engine, pipeline_id, config.SMOKE_TEST_DISTILL_CASE)
    if request.distill is None:
        raise ValueError(f"Pipeline {pipeline_id} has no `distill` Stage")
    registered = upload_dataset_version(state.engine, state.object_store, request.name, path)
    return dataset_reference(request.name, registered["version"])
