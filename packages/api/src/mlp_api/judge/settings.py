from pydantic import Field, model_validator
from sqlalchemy import JSON, Boolean, Column, Engine, Integer, Table, Text, delete, insert, select

from mlp_api.storage.database import metadata
from mlp_core import config
from mlp_core.endpoint_spec import EndpointSpec, without_stored_quantization
from mlp_core.pipeline_request.schema import Strict

# One row at most: the platform has one Judge.
judge = Table(
    "judge",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("enabled", Boolean, nullable=False),
    # What the Judge serves, as the Endpoint start form sends it, unpinned.
    Column("spec", JSON),
    Column("rubric", Text, nullable=False),
)


class JudgeSettings(Strict):
    enabled: bool = False
    spec: EndpointSpec | None = None
    rubric: str = Field(config.JUDGE_RUBRIC, min_length=1)

    @model_validator(mode="after")
    def check_spec(self) -> "JudgeSettings":
        if self.enabled and self.spec is None:
            raise ValueError("The Judge needs a `spec` naming the model it serves")
        return self


def judge_settings(engine: Engine) -> JudgeSettings:
    """The stored settings; judging is off with the default rubric until they are saved."""
    with engine.connect() as connection:
        row = connection.execute(select(judge)).first()
    if row is None:
        return JudgeSettings()
    spec = row.spec and without_stored_quantization(row.spec)
    return JudgeSettings(enabled=row.enabled, spec=spec, rubric=row.rubric)


def store_judge_settings(engine: Engine, settings: JudgeSettings) -> None:
    spec = settings.spec and settings.spec.model_dump(mode="json", exclude_defaults=True)
    with engine.begin() as connection:
        connection.execute(delete(judge))
        connection.execute(
            insert(judge).values(enabled=settings.enabled, spec=spec, rubric=settings.rubric)
        )
