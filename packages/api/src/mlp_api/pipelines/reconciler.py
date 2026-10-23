import logging
import threading
from datetime import UTC, datetime

from sqlalchemy import Engine, select

from mlp_api.endpoints.history_retention import trim_endpoint_history
from mlp_api.endpoints.lifecycle import reconcile_endpoints
from mlp_api.endpoints.stats_history import sample_endpoint_stats
from mlp_api.pipelines.lifecycle import pipeline, pipeline_secret_name, set_pipeline, unfinished
from mlp_api.pipelines.mlflow_run import log_pipeline_request
from mlp_api.smoke_tests.cases import case_results, has_pending_serving_cases
from mlp_api.smoke_tests.lifecycle import clean_up_smoke_tests
from mlp_api.smoke_tests.serving_cases import run_serving_cases
from mlp_core import config

logger = logging.getLogger(__name__)


def reconcile_forever(state, stop: threading.Event) -> None:
    """Reconciles every RECONCILE_INTERVAL until `stop` is set; a failed round is retried."""
    while not stop.wait(config.RECONCILE_INTERVAL.total_seconds()):
        try:
            reconcile_once(state)
        except Exception:
            logger.exception("Reconciling failed")


def reconcile_once(state) -> None:
    """One round: Pipelines and Endpoints follow the cluster and are sampled; Smoke Tests go on."""
    now = datetime.now(UTC)
    reconcile_pipelines(state)
    reconcile_endpoints(state.engine, state.cluster)
    sample_endpoint_stats(state, now)
    trim_endpoint_history(state, now)
    run_serving_cases(state)
    clean_up_smoke_tests(state)


def reconcile_pipelines(state) -> None:
    """Copies Kubeflow run and step states into the Pipelines and deletes unneeded Secrets."""
    engine, cluster = state.engine, state.cluster
    # 1. The status of every unfinished Pipeline, and a Smoke Test's case results, from its run.
    for row in unfinished_pipelines(engine):
        if row.kubeflow_run_id is None:
            continue
        run = cluster.find_run(row.kubeflow_run_id)
        status = "failed" if run is None else config.PIPELINE_STATUSES.get(run.state, "pending")
        finished = status in config.FINISHED_STATUSES
        # A Smoke Test runs on while its serving cases' Endpoints, outside the run, have no result.
        if finished and row.cases is not None and has_pending_serving_cases(row.cases):
            status, finished = "running", False
        values = {"status": status}
        if row.cases is not None:
            values["cases"] = case_results(row.cases, run, finished)
        # The request goes to the MLflow Run when KFP first reports it, retried while unfinished.
        url = run and run.mlflow_run_url
        first_seen = url and row.mlflow_run_url is None
        if first_seen and (logged_pipeline_request(state, row, url) or finished):
            values["mlflow_run_url"] = url
        waiting = run is not None and not finished
        if waiting and cluster.is_waiting_for_gpu(row.kubeflow_run_id, row.id):
            values["status"] = config.WAITING_FOR_GPU
        set_pipeline(engine, row.id, **values)

    # 2. Secrets of finished or unknown Pipelines, and the backstop for old ones. Listed
    # before the Pipelines, since a Pipeline is stored before its Secret is created.
    secret_ages = cluster.secret_ages()
    needed = {pipeline_secret_name(row.id) for row in unfinished_pipelines(engine)}
    for name, age in secret_ages.items():
        if name not in needed or age > config.SECRET_MAX_AGE:
            cluster.delete_secret(name)

    # 3. TrainJobs of finished or unknown Pipelines, whose launcher a KFP terminate killed hard.
    running = {row.id for row in unfinished_pipelines(engine)}
    for pipeline_id in cluster.trainjob_pipelines() - running:
        cluster.delete_trainjobs(pipeline_id)


def logged_pipeline_request(state, row, mlflow_run_url: str) -> bool:
    try:
        log_pipeline_request(state.model_registry, state.object_store, mlflow_run_url, row.request)
    except Exception:
        logger.exception("Logging Pipeline %s's request to MLflow failed", row.id)
        return False
    return True


def unfinished_pipelines(engine: Engine) -> list:
    query = select(pipeline).where(unfinished)
    with engine.connect() as connection:
        return connection.execute(query).all()
