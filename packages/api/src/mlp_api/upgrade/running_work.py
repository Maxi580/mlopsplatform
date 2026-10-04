import os

from sqlalchemy import Engine, create_engine, select

from mlp_api.endpoints.lifecycle import stop_endpoint
from mlp_api.endpoints.table import endpoint, not_stopped
from mlp_api.pipelines.cluster import Cluster
from mlp_api.pipelines.lifecycle import cancel_pipeline, pipeline, unfinished


def print_running_work() -> None:
    """`running-work`: one line per unfinished Pipeline and running Endpoint, for install.sh."""
    engine = create_engine(os.environ["DATABASE_URL"])
    pipelines, endpoints = find_running_work(engine)
    for row in pipelines:
        print(f"Pipeline {row.id} {row.name} ({row.status})")
    for row in endpoints:
        print(f"Endpoint {row.name} ({row.status})")
    engine.dispose()


def stop_running_work() -> None:
    """`stop-running-work`: cancels every unfinished Pipeline and stops every Endpoint."""
    engine, cluster = create_engine(os.environ["DATABASE_URL"]), Cluster()
    pipelines, endpoints = find_running_work(engine)
    for row in pipelines:
        # One still being submitted has no run yet; the restarted API marks it failed.
        if row.kubeflow_run_id is not None:
            cancel_pipeline(engine, cluster, row.id)
    for row in endpoints:
        stop_endpoint(engine, cluster, row.name)
    engine.dispose()


def find_running_work(engine: Engine) -> tuple[list, list]:
    """The unfinished Pipelines and the Endpoints that aren't stopped, oldest first."""
    with engine.connect() as connection:
        pipelines = connection.execute(select(pipeline).where(unfinished).order_by(pipeline.c.id))
        endpoints = connection.execute(select(endpoint).where(not_stopped).order_by(endpoint.c.id))
        return pipelines.all(), endpoints.all()
