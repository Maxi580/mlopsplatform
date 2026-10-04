from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from mlp_core.config import (
    ALGORITHMS,
    BACKENDS,
    BENCHMARKS,
    DISTILL_OUTPUT,
    FINETUNE_OUTPUT,
    MAX_LORA_RANK,
)
from mlp_core.endpoint_spec import ENDPOINT_NAME_PATTERN, EndpointName, ServingOptions
from mlp_core.pipeline_request.references import (
    MODEL_NAME_PATTERN,
    BaseModelReference,
    DatasetReference,
    ModelReference,
)

METHODS = tuple(sorted({method for methods in BACKENDS.values() for method in methods}))
EndpointReference = Annotated[str, Field(pattern=f"^endpoint:{ENDPOINT_NAME_PATTERN}$")]
InClusterTeacher = TypeAdapter(BaseModelReference | ModelReference | EndpointReference)


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
    dataset: DatasetReference | Literal[DISTILL_OUTPUT]
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


class ToolFunction(Strict):
    name: str = Field(pattern=r"^[\w-]{1,64}$")
    description: str | None = None
    # The JSON Schema of the arguments; validation checks that it is one.
    parameters: dict[str, Any] = {"type": "object", "properties": {}}


class Tool(Strict):
    """A function the Teacher may call, in the OpenAI `tools` format."""

    type: Literal["function"] = "function"
    function: ToolFunction


class Distill(Strict):
    """A Distillation Dataset: each prompt's single Teacher reply, a text or tool calls (#17)."""

    # A Dataset of `prompt` rows.
    dataset: DatasetReference
    # A Base Model or Model Version run here, or a running Endpoint; with `api_url`, the name of
    # the API's model.
    teacher: str
    # An OpenAI-compatible API serving the Teacher, e.g. https://api.openai.com/v1; its key is the
    # `teacher_api_key` Secret.
    api_url: str | None = Field(None, pattern=r"^https?://\S+$", title="API URL")
    # Offered with every prompt; a reply may call them.
    tools: list[Tool] | None = None
    # Off: the Teacher is asked for one call per reply, and replies with several are dropped.
    parallel_tool_calls: bool = False
    max_tokens: int | None = Field(None, gt=0)
    temperature: float | None = Field(None, ge=0)
    # For the vLLM a Base Model or Model Version Teacher runs on.
    serving: ServingOptions | None = None

    @model_validator(mode="after")
    def check_teacher(self) -> "Distill":
        if self.api_url is None:
            try:
                InClusterTeacher.validate_python(self.teacher)
            except ValidationError:
                raise ValueError(
                    "`teacher` is hf:…, model:… or endpoint:…, or a model at `api_url`"
                ) from None
        if self.serving and (self.api_url or self.teacher.startswith("endpoint:")):
            raise ValueError("this Teacher serves with its own options; leave out `serving`")
        return self


class Evaluate(Strict):
    """Benchmarks from the catalog, run against a model or a running Endpoint."""

    # Validation names `@finetune`, the Pipeline's last Model Version, when none is given.
    model: (
        BaseModelReference | ModelReference | EndpointReference | Literal[FINETUNE_OUTPUT] | None
    ) = None
    benchmarks: list[Literal[tuple(BENCHMARKS)]] = Field(min_length=1)
    # Samples per task, for a quick look; scores on fewer samples don't compare with full runs.
    limit: int | None = Field(None, gt=0)
    # For the vLLM `evaluate` starts; an Endpoint serves with its own.
    serving: ServingOptions | None = None


class Serve(ServingOptions):
    """An Endpoint for the Pipeline's last Model Version, started once the Pipeline made it."""

    # Validation pins the Pipeline's name when none is given.
    name: EndpointName | None = Field(None, title="Endpoint name")


class PipelineRequest(Strict):
    schema_version: Literal[1] = 1
    name: str = Field(pattern=f"^{MODEL_NAME_PATTERN}$", max_length=63)
    distill: Distill | None = None
    finetune: Finetune | None = None
    evaluate: Evaluate | None = None
    serve: Serve | None = None

    @model_validator(mode="after")
    def check_stages(self) -> "PipelineRequest":
        if not (self.distill or self.finetune or self.evaluate):
            raise ValueError("enable at least one of `distill`, `finetune` and `evaluate`")
        if self.serve and not self.finetune:
            raise ValueError("`serve` serves the output of `finetune`, which isn't enabled")
        return self
