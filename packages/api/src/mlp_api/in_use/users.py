import json

from sqlalchemy import Engine, select

from mlp_api.endpoints.table import endpoint, not_stopped
from mlp_api.pipelines.lifecycle import pipeline, unfinished


def refuse_while_in_use(engine: Engine, reference: str, produced_by: str | None = None) -> None:
    """ValueError naming every unfinished Pipeline and Endpoint that uses the Reference."""
    users = users_of(engine, reference, produced_by)
    if users:
        raise ValueError(f"{reference} is used by {', '.join(users)}")


def users_of(engine: Engine, reference: str, produced_by: str | None = None) -> list[str]:
    """Unfinished Pipelines naming, reusing or making the Reference, and Endpoints serving it."""
    with engine.connect() as connection:
        pipelines = connection.execute(select(pipeline).where(unfinished)).all()
        endpoints = connection.execute(select(endpoint).where(not_stopped)).all()
    # A resolved request pins every Reference, so it holds the exact string as one JSON value;
    # a resume names the Model Versions, Dataset Version and Checkpoint it reuses the same way.
    return [
        f"Pipeline {row.name} (#{row.id})"
        for row in pipelines
        if str(row.id) == produced_by
        or json.dumps(reference) in json.dumps([row.request, row.resume])
    ] + [f"Endpoint {row.name}" for row in endpoints if reference in row.uses]
