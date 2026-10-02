from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from mlp_core.config import BACKENDS
from mlp_core.pipeline_request.references import BaseModelReference, DatasetReference

METHODS = tuple(sorted({method for methods in BACKENDS.values() for method in methods}))


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrainerSettings(BaseModel):
    """Requires the basic values and allows every other field of its TRL/PEFT config class."""

    model_config = ConfigDict(extra="allow")
    trainer_config: ClassVar[str]


class SftSettings(TrainerSettings):
    trainer_config: ClassVar[str] = "SFTConfig"

    learning_rate: float
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_length: int


class LoraSettings(TrainerSettings):
    trainer_config: ClassVar[str] = "LoraConfig"

    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: str | list[str]


class SftPhase(Strict):
    algorithm: Literal["sft"]
    dataset: DatasetReference
    method: Literal[METHODS]
    settings: SftSettings
    lora: LoraSettings


class Finetune(Strict):
    base_model: BaseModelReference
    backend: Literal[tuple(BACKENDS)]
    phases: list[SftPhase] = Field(min_length=1, max_length=1)


class PipelineRequest(Strict):
    schema_version: Literal[1] = 1
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]*$", max_length=63)
    finetune: Finetune
