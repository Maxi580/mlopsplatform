from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mlp_core.trainers import (
    ALGORITHMS,
    BACKENDS,
    LORA_CONFIG,
    METHODS,
    trainer_config_schema,
)


def published_schema(config: str) -> dict:
    schema = trainer_config_schema(config)
    if schema is None:
        return {}
    # Validation rejects trust_remote_code anywhere, so the schema doesn't offer it.
    properties = {
        name: field for name, field in schema["properties"].items() if name != "trust_remote_code"
    }
    return {**schema, "properties": properties}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SftPhase(Strict):
    algorithm: Literal["sft"]
    dataset: str = Field(pattern=r"^dataset:[\w.-]+(@\d+)?$")
    method: Literal[tuple(METHODS)] = "lora"
    settings: dict[str, Any] = Field(
        default_factory=dict,
        json_schema_extra=lambda schema: schema.update(published_schema(ALGORITHMS["sft"])),
    )
    lora: dict[str, Any] = Field(
        default_factory=dict,
        json_schema_extra=lambda schema: schema.update(published_schema(LORA_CONFIG)),
    )


class Finetune(Strict):
    base_model: str = Field(pattern=r"^hf:[\w.-]+/[\w.-]+(@[\w.-]+)?$")
    backend: Literal[tuple(BACKENDS)] = "hf"
    phases: list[SftPhase] = Field(min_length=1, max_length=1)


class PipelineRequest(Strict):
    schema_version: Literal[1] = 1
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]*$", max_length=63)
    finetune: Finetune
