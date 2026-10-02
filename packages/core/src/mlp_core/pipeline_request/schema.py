from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mlp_core.config import ALGORITHMS, BACKENDS

METHODS = tuple(sorted({method for methods in BACKENDS.values() for method in methods}))


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Settings classes require the basic values and allow every other TRL/PEFT field.
class SftSettings(BaseModel):
    model_config = ConfigDict(extra="allow")

    learning_rate: float
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_length: int


class LoraSettings(BaseModel):
    model_config = ConfigDict(extra="allow")

    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: str | list[str]


class SftPhase(Strict):
    algorithm: Literal[tuple(ALGORITHMS)]
    dataset: str = Field(pattern=r"^dataset:[\w.-]+(@\d+)?$")
    method: Literal[METHODS]
    settings: SftSettings
    lora: LoraSettings


class Finetune(Strict):
    base_model: str = Field(pattern=r"^hf:[\w.-]+/[\w.-]+(@[\w.-]+)?$")
    backend: Literal[tuple(BACKENDS)]
    phases: list[SftPhase] = Field(min_length=1, max_length=1)


class PipelineRequest(Strict):
    schema_version: Literal[1] = 1
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]*$", max_length=63)
    finetune: Finetune
