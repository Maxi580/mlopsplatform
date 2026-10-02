import json
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Engine,
    Integer,
    String,
    Table,
    insert,
    select,
    update,
)

from mlp_api.database import metadata
from mlp_api.pipelines.cluster import Cluster
from mlp_api.pipelines.compiler import compile_pipeline
from mlp_core import config
from mlp_core.pipeline_request.schema import PipelineRequest
from mlp_core.settings import Settings

pipeline = Table(
    "pipeline",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String, nullable=False),
    Column("owner", String, nullable=False),
    # The resolved request; it never holds a Secret value.
    Column("request", JSON, nullable=False),
    Column("status", String, nullable=False),
    Column("kubeflow_run_id", String),
    Column("mlflow_run_url", String),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
unfinished = pipeline.c.status.not_in(config.FINISHED_STATUSES)


def submit_pipeline(
    engine: Engine,
    cluster: Cluster,
    settings: Settings,
    request: PipelineRequest,
    secrets: dict[str, str],
) -> int:
    """The new Pipeline's ID, once its Kubeflow run is submitted; RuntimeError if that failed."""
    # 1. The Pipeline, whose ID names its Secret.
    with engine.begin() as connection:
        pipeline_id = connection.execute(
            insert(pipeline).values(
                name=request.name,
                owner=config.OWNER,
                request=request.model_dump(mode="json"),
                status="pending",
                created_at=datetime.now(UTC),
            )
        ).inserted_primary_key[0]

    # 2. The Secret and the run; a failure leaves the Secret to the reconciler.
    try:
        cluster.create_secret(pipeline_secret_name(pipeline_id), secrets)
        spec = compile_pipeline(pipeline_id, request, cluster.steps, settings.gpus_per_stage)
        run_id = cluster.submit_run(f"{request.name}-{pipeline_id}", spec)
    except Exception as error:
        set_pipeline(engine, pipeline_id, status="failed")
        raise RuntimeError(f"Kubeflow did not start Pipeline {pipeline_id}: {error}") from None
    set_pipeline(engine, pipeline_id, kubeflow_run_id=run_id)
    return pipeline_id


def cancel_pipeline(engine: Engine, cluster: Cluster, pipeline_id: int) -> None:
    """Stops the run and deletes the Secret; LookupError if unknown, ValueError if it can't."""
    row = find_pipeline(engine, pipeline_id)
    if row is None:
        raise LookupError(f"No Pipeline {pipeline_id}")
    if row.status in config.FINISHED_STATUSES:
        raise ValueError(f"Pipeline {pipeline_id} already {row.status}")
    if row.kubeflow_run_id is None:
        raise ValueError(f"Pipeline {pipeline_id} is still being submitted; try again")
    cluster.terminate_run(row.kubeflow_run_id)
    cluster.delete_secret(pipeline_secret_name(pipeline_id))
    set_pipeline(engine, pipeline_id, status="cancelled")


def list_pipelines(engine: Engine) -> list[dict]:
    """Every Pipeline, newest first, with its enabled Stages and links."""
    with engine.connect() as connection:
        rows = connection.execute(select(pipeline).order_by(pipeline.c.id.desc())).all()
    return [
        {
            "id": row.id,
            "name": row.name,
            "owner": row.owner,
            "status": row.status,
            "stages": [stage for stage in config.STAGES if stage in row.request],
            "created_at": row.created_at.isoformat(),
            "kubeflow_run_url": row.kubeflow_run_id
            and config.KUBEFLOW_RUN_URL.format(run_id=row.kubeflow_run_id),
            "mlflow_run_url": row.mlflow_run_url,
        }
        for row in rows
    ]


def refuse_while_in_use(engine: Engine, reference: str, produced_by: str | None = None) -> None:
    """ValueError naming every unfinished Pipeline whose request names the Reference or makes it."""
    with engine.connect() as connection:
        rows = connection.execute(select(pipeline).where(unfinished)).all()
    # A resolved request pins every Reference, so it holds the exact string as one JSON value.
    using = [
        f"{row.name} (#{row.id})"
        for row in rows
        if str(row.id) == produced_by or json.dumps(reference) in json.dumps(row.request)
    ]
    if using:
        raise ValueError(f"{reference} is used by Pipeline {', '.join(using)}")


# Submission happens inside a request, so after a restart no unsubmitted Pipeline is in flight.
def fail_unsubmitted_pipelines(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            update(pipeline)
            .where(pipeline.c.kubeflow_run_id.is_(None), unfinished)
            .values(status="failed")
        )


def pipeline_secret_name(pipeline_id: int) -> str:
    return config.PIPELINE_SECRET_NAME.format(id=pipeline_id)


# Only an unfinished Pipeline changes, so a cancel is never overwritten by a slower reconcile.
def set_pipeline(engine: Engine, pipeline_id: int, **values) -> None:
    with engine.begin() as connection:
        connection.execute(
            update(pipeline).where(pipeline.c.id == pipeline_id, unfinished).values(**values)
        )


def find_pipeline(engine: Engine, pipeline_id: int):
    with engine.connect() as connection:
        return connection.execute(select(pipeline).where(pipeline.c.id == pipeline_id)).first()
