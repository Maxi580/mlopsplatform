from collections import Counter

from sqlalchemy import Engine, select

from mlp_api.in_use.users import refuse_while_in_use
from mlp_api.pipelines.lifecycle import pipeline, unfinished
from mlp_api.storage.object_store import ObjectStore
from mlp_core import config
from mlp_core.pipeline_request.references import checkpoint_prefix


def list_checkpoints(engine: Engine, object_store: ObjectStore) -> list[dict]:
    """Every Checkpoint a finished Pipeline kept, with its Phase index and size, oldest first."""
    sizes = Counter()
    for item in object_store.list_objects(object_store.bucket, config.CHECKPOINTS_PREFIX):
        pipeline_id, phase_index = (
            item["Key"].removeprefix(config.CHECKPOINTS_PREFIX).split("/")[:2]
        )
        sizes[int(pipeline_id), int(phase_index)] += item["Size"]
    # A running Pipeline's Checkpoints are still its own.
    query = select(pipeline.c.id, pipeline.c.name).where(unfinished)
    with engine.connect() as connection:
        running = dict(connection.execute(query).all())
        names = dict(connection.execute(select(pipeline.c.id, pipeline.c.name)).all())
    return [
        {
            "pipeline_id": pipeline_id,
            "pipeline_name": names.get(pipeline_id),
            "phase_index": phase_index,
            "size_bytes": size,
        }
        for (pipeline_id, phase_index), size in sorted(sizes.items())
        if pipeline_id not in running
    ]


def delete_checkpoint(
    engine: Engine, object_store: ObjectStore, pipeline_id: int, phase_index: int
) -> None:
    """Deletes the Checkpoint; LookupError if there is none, ValueError while it is needed."""
    prefix = checkpoint_prefix(pipeline_id, phase_index)
    if not any(object_store.list_objects(object_store.bucket, prefix)):
        raise LookupError(
            f"Pipeline {pipeline_id} kept no Checkpoint for Phase index {phase_index}"
        )
    # Needed by its own Pipeline while that runs, and by a running resume of it.
    refuse_while_in_use(engine, prefix, produced_by=str(pipeline_id))
    object_store.delete_all(object_store.bucket, prefix)
