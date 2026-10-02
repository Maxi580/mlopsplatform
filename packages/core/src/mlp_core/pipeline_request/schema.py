from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlp_core.config import ALGORITHMS, BACKENDS, MAX_LORA_RANK
from mlp_core.endpoint_spec import EndpointName, ServingOptions
from mlp_core.pipeline_request.references import (
    MODEL_NAME_PATTERN,
    BaseModelReference,
    DatasetReference,
    ModelReference,
)

METHODS = tuple(sorted({method for methods in BACKENDS.values() for method in methods}))


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrainerSettings(BaseModel):
    """Requires the basic values and allows every other field of its TRL/PEFT config class."""

    model_config = ConfigDict(extra="allow")


# Checked against the TRL config of the Phase's algorithm.
class PhaseSettings(TrainerSettings):
    learning_rate: float
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_length: int


# Checked against PEFT's LoraConfig.
class LoraSettings(TrainerSettings):
    r: int = Field(gt=0, le=MAX_LORA_RANK)
    lora_alpha: int
    lora_dropout: float
    target_modules: str | list[str]


class Phase(Strict):
    algorithm: Literal[tuple(ALGORITHMS)]
    dataset: DatasetReference
    method: Literal[METHODS]
    settings: PhaseSettings
    lora: LoraSettings


class Finetune(Strict):
    # `from` is a Python keyword, so the field has another name and is read and written as `from`.
    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    # The first Phase starts from a Base Model or from a full-weight Model Version.
    base_model: BaseModelReference | None = None
    from_: ModelReference | None = Field(None, alias="from", title="From Model Version")
    backend: Literal[tuple(BACKENDS)]
    phases: list[Phase] = Field(min_length=1, max_length=1)

    @model_validator(mode="after")
    def check_one_starting_model(self) -> "Finetune":
        if (self.base_model is None) == (self.from_ is None):
            raise ValueError("name exactly one of `base_model` and `from`")
        return self

    @property
    def starting_model(self) -> str:
        return self.base_model or self.from_


class Serve(ServingOptions):
    """An Endpoint for the Pipeline's last Model Version, started once the Pipeline made it."""

    # Validation pins the Pipeline's name when none is given.
    name: EndpointName | None = Field(None, title="Endpoint name")


class PipelineRequest(Strict):
    schema_version: Literal[1] = 1
    name: str = Field(pattern=f"^{MODEL_NAME_PATTERN}$", max_length=63)
    finetune: Finetune
    serve: Serve | None = None
