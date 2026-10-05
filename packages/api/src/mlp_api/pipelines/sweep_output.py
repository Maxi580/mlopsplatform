from sqlalchemy import Engine

from mlp_api.pipelines.lifecycle import find_running_request, set_pipeline
from mlp_core import config


def record_sweep_output(engine: Engine, pipeline_id: int, best: dict) -> None:
    """Stores the best parameters and objective of the Pipeline's `sweep` step on it."""
    request = find_running_request(engine, pipeline_id, config.SMOKE_TEST_SWEEP_CASE)
    if request.sweep is None:
        raise ValueError(f"Pipeline {pipeline_id} has no `sweep` Stage")
    set_pipeline(engine, pipeline_id, sweep=best)
