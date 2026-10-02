import logging
import threading

from sqlalchemy import Engine, select

from mlp_api.pipelines.cluster import Cluster
from mlp_api.pipelines.lifecycle import pipeline, pipeline_secret_name, set_pipeline, unfinished
from mlp_core import config

logger = logging.getLogger(__name__)


def reconcile_forever(state, stop: threading.Event) -> None:
    """Reconciles every RECONCILE_INTERVAL until `stop` is set; a failed round is retried."""
    while not stop.wait(config.RECONCILE_INTERVAL.total_seconds()):
        try:
            reconcile_pipelines(state.engine, state.cluster)
        except Exception:
            logger.exception("Reconciling Pipelines failed")


def reconcile_pipelines(engine: Engine, cluster: Cluster) -> None:
    """Copies Kubeflow run states into the Pipelines and deletes Secrets no Pipeline needs."""
    # 1. The status of every unfinished Pipeline, from its Kubeflow run.
    for row in unfinished_pipelines(engine):
        if row.kubeflow_run_id is None:
            continue
        run = cluster.find_run(row.kubeflow_run_id)
        if run is None:
            set_pipeline(engine, row.id, status="failed")
            continue
        status = config.PIPELINE_STATUSES.get(run.state, "pending")
        if status not in config.FINISHED_STATUSES and cluster.is_waiting_for_gpu(
            row.kubeflow_run_id
        ):
            status = config.WAITING_FOR_GPU
        set_pipeline(engine, row.id, status=status, mlflow_run_url=run.mlflow_run_url)

    # 2. Secrets of finished or unknown Pipelines, and the backstop for old ones. Listed
    # before the Pipelines, since a Pipeline is stored before its Secret is created.
    secret_ages = cluster.secret_ages()
    needed = {pipeline_secret_name(row.id) for row in unfinished_pipelines(engine)}
    for name, age in secret_ages.items():
        if name not in needed or age > config.SECRET_MAX_AGE:
            cluster.delete_secret(name)


def unfinished_pipelines(engine: Engine) -> list:
    query = select(pipeline).where(unfinished)
    with engine.connect() as connection:
        return connection.execute(query).all()
