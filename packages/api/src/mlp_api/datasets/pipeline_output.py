from pathlib import Path

from sqlalchemy import Engine, select

from mlp_api.datasets.registry import dataset_version, upload_dataset_version
from mlp_api.pipelines.lifecycle import find_running_request
from mlp_core import config
from mlp_core.pipeline_request.references import dataset_reference


def register_distillation_dataset(state, pipeline_id: int, path: Path) -> str:
    """The `dataset:` Reference of the Pipeline's `distill` output, registered under its name."""
    request = find_running_request(state.engine, pipeline_id, config.SMOKE_TEST_DISTILL_CASE)
    if request.distill is None:
        raise ValueError(f"Pipeline {pipeline_id} has no `distill` Stage")
    registered = upload_dataset_version(
        state.engine, state.object_store, request.name, path, pipeline_id
    )
    return dataset_reference(request.name, registered["version"])


def find_distillation_dataset(engine: Engine, pipeline_id: int) -> str | None:
    """The `dataset:` Reference the Pipeline's `distill` Stage registered, if it still exists."""
    query = select(dataset_version).where(
        dataset_version.c.pipeline == pipeline_id, ~dataset_version.c.deleted
    )
    with engine.connect() as connection:
        row = connection.execute(query).first()
    return row and dataset_reference(row.name, row.version)
