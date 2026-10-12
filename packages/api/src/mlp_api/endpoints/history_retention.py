import logging
import uuid
from datetime import datetime

from mlp_api.endpoints.stats_history import round_ms
from mlp_api.models.mlflow import MLflow
from mlp_core import config

logger = logging.getLogger(__name__)


def trim_endpoint_history(state, now: datetime) -> None:
    """Deletes every Endpoint Stats History older than its retention, at most once an interval."""
    # 1. Once an interval; a failure waits for the next one.
    trimmed_at = state.endpoint_history_trimmed_at
    if trimmed_at and now - trimmed_at < config.ENDPOINT_HISTORY_TRIM_INTERVAL:
        return
    state.endpoint_history_trimmed_at = now

    # 2. Each Endpoint name's older traces, and its day Runs that end before the window.
    registry: MLflow = state.model_registry
    start = now - config.ENDPOINT_HISTORY_RETENTION
    first_day = start.strftime(config.ENDPOINT_HISTORY_RUN_NAME)
    try:
        prefix = config.ENDPOINT_HISTORY_EXPERIMENT.format(name="")
        for experiment_id in registry.experiment_ids(prefix):
            registry.delete_traces(experiment_id, round_ms(start))
            for day, run_id in registry.run_ids(experiment_id).items():
                if day < first_day:
                    registry.delete_run(run_id)
    except Exception:
        logger.exception("Trimming the Endpoint Stats History failed")


def delete_endpoint_history(state, name: str, now: datetime) -> None:
    """Deletes the Endpoint name's Stats History; if MLflow fails, the window empties it later."""
    # 1. Its cached reading and day Run.
    state.server_readings.pop(name, None)
    state.endpoint_day_runs.pop(name, None)

    # 2. Its traces for good; its experiment renamed, freeing the name at once, then deleted.
    registry: MLflow = state.model_registry
    experiment = config.ENDPOINT_HISTORY_EXPERIMENT.format(name=name)
    deleted = config.ENDPOINT_HISTORY_DELETED_EXPERIMENT.format(
        experiment=experiment, uuid=uuid.uuid4().hex
    )
    try:
        experiment_id = registry.experiment_id(experiment, create=False)
        if experiment_id is None:
            return
        registry.delete_traces(experiment_id, round_ms(now))
        registry.rename_experiment(experiment_id, deleted)
        registry.delete_experiment(experiment_id)
    except Exception:
        logger.exception("Deleting Endpoint %s's Stats History failed", name)
