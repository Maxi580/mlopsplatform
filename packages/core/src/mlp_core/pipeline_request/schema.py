from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_serializer,
    model_validator,
)

from mlp_core.config import (
    ALGORITHMS,
    ASSISTANT_ONLY_LOSS_INFOBOX,
    BACKENDS,
    BENCHMARKS,
    CALIBRATION_DATASET,
    CALIBRATION_MAX_LENGTH,
    CALIBRATION_SAMPLES,
    DISTILL_OUTPUT,
    FINETUNE_OUTPUT,
    PERFORMANCE_CONCURRENCY,
    PERFORMANCE_OUTPUT_TOKENS,
    PERFORMANCE_PROMPT_TOKENS,
    PERFORMANCE_REQUESTS,
    QUANTIZATION_SCHEMES,
    QUANTIZE_IGNORE,
    QUANTIZE_OUTPUT,
    REWARD_SOURCE_MAX_LENGTH,
    REWARDS_INFOBOX,
    UNCALIBRATED_SCHEMES,
    WEIGHT_METHODS,
)
from mlp_core.endpoint_spec import ENDPOINT_NAME_PATTERN, EndpointName, ServingOptions
from mlp_core.pipeline_request.references import (
    MODEL_NAME_PATTERN,
    BaseModelReference,
    DatasetReference,
    ModelReference,
)

EndpointReference = Annotated[str, Field(pattern=f"^endpoint:{ENDPOINT_NAME_PATTERN}$")]
InClusterTeacher = TypeAdapter(BaseModelReference | ModelReference | EndpointReference)
JobTeacher = TypeAdapter(BaseModelReference | ModelReference)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrainerSettings(BaseModel):
    """Requires the basic values and allows every other field of its TRL/PEFT config class."""

    model_config = ConfigDict(extra="allow")


# Checked against the TRL config of the Phase's algorithm, which also requires its length setting.
class PhaseSettings(TrainerSettings):
    learning_rate: float
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    # An SFTConfig setting, a field of its own for its infobox; validation checks the rest.
    assistant_only_loss: bool | None = Field(
        None, title="Assistant-only loss (sft)", description=ASSISTANT_ONLY_LOSS_INFOBOX
    )

    # Left out, it reaches no TRL config, as only SFTConfig has it.
    @model_serializer(mode="wrap")
    def without_unset_assistant_only_loss(self, serialize) -> dict:
        settings = serialize(self)
        if settings.get("assistant_only_loss") is None:
            settings.pop("assistant_only_loss", None)
        return settings


# Checked against PEFT's LoraConfig.
class LoraSettings(TrainerSettings):
    r: int = Field(gt=0)
    lora_alpha: int
    lora_dropout: float
    target_modules: str | list[str]


class Reward(Strict):
    weight: float
    # Defines `def reward(sample, item)`; runs only in the Sandbox (#19).
    source: str = Field(max_length=REWARD_SOURCE_MAX_LENGTH, json_schema_extra={"format": "python"})


# Names its rewards/<name>/… metrics.
RewardName = Annotated[str, Field(pattern=r"^[\w-]{1,64}$")]


class Phase(Strict):
    algorithm: Literal[tuple(ALGORITHMS)]
    dataset: DatasetReference | Literal[DISTILL_OUTPUT]
    method: Literal[WEIGHT_METHODS] = "lora"
    # `merged` registers the Adapter merged into its base, as full weights; `full` ignores it.
    output: Literal["adapter", "merged"] = "adapter"
    settings: PhaseSettings
    # A new Adapter's settings; after a kept Adapter, a Phase continues it as it is (#19).
    lora: LoraSettings | None = None
    # The Base Model or Model Version a `distillation` Phase learns the token probabilities of.
    teacher: str | None = None
    rewards: dict[RewardName, Reward] | None = Field(None, description=REWARDS_INFOBOX)

    @model_validator(mode="after")
    def check_rewards(self) -> "Phase":
        if not ALGORITHMS[self.algorithm]["learns_from_rewards"]:
            if self.rewards is not None:
                raise ValueError(f"a {self.algorithm} Phase has no `rewards`")
        elif not self.rewards:
            raise ValueError(f"a {self.algorithm} Phase learns from `rewards`; name at least one")
        return self

    @model_validator(mode="after")
    def check_teacher(self) -> "Phase":
        if not ALGORITHMS[self.algorithm]["learns_from_teacher"]:
            if self.teacher:
                raise ValueError(f"a {self.algorithm} Phase learns from no `teacher`")
            return self
        if self.teacher is None:
            raise ValueError(f"name the `teacher` a {self.algorithm} Phase learns from")
        try:
            JobTeacher.validate_python(self.teacher)
        except ValidationError:
            raise ValueError(
                "the Teacher's token probabilities need it loaded beside the Student: hf:… or "
                "model:…, not an API or Endpoint"
            ) from None
        return self

    @property
    def keeps_adapter(self) -> bool:
        return self.method != "full" and self.output == "adapter"

    @property
    def merges_adapter(self) -> bool:
        return self.method != "full" and self.output == "merged"


class Finetune(Strict):
    # `from` is a Python keyword, so the field has another name and is read and written as `from`.
    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    # The first Phase starts from a Base Model or from a full-weight Model Version.
    base_model: BaseModelReference | None = None
    from_: ModelReference | None = Field(None, alias="from", title="From Model Version")
    backend: Literal[tuple(BACKENDS)] = "hf"
    # Run in order, each starting from the Model Version the one before it registered.
    phases: list[Phase] = Field(min_length=1)

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


class Calibration(Strict):
    """The rows a scheme measures activations on, to choose its scales."""

    # A Dataset of messages or text rows; the install registers the default one.
    dataset: DatasetReference = f"dataset:{CALIBRATION_DATASET}"
    samples: int = Field(CALIBRATION_SAMPLES, gt=0)
    # Tokens per sample; longer rows are cut.
    max_length: int = Field(CALIBRATION_MAX_LENGTH, gt=0)


class Quantize(Strict):
    """A quantized copy of a model by llm-compressor, registered as full weights."""

    # Validation names `@finetune` when none is given; an Adapter is merged into its base first.
    model: BaseModelReference | ModelReference | Literal[FINETUNE_OUTPUT] | None = None
    scheme: Literal[tuple(QUANTIZATION_SCHEMES)]
    # Layers kept unquantized.
    ignore: list[str] = QUANTIZE_IGNORE
    calibration: Calibration | None = None

    @model_validator(mode="after")
    def check_calibration(self) -> "Quantize":
        calibrated = self.scheme not in UNCALIBRATED_SCHEMES
        if calibrated and self.calibration is None:
            raise ValueError(
                f"{self.scheme} calibrates on Dataset rows; add `calibration`, which `{{}}` "
                "fills with the default calibration Dataset"
            )
        if not calibrated and self.calibration is not None:
            raise ValueError(f"{self.scheme} needs no `calibration`; leave it out")
        return self


class Performance(Strict):
    """Serving performance by GuideLLM: synthetic chat requests, a fixed number at once."""

    prompt_tokens: int = Field(PERFORMANCE_PROMPT_TOKENS, gt=0)
    output_tokens: int = Field(PERFORMANCE_OUTPUT_TOKENS, gt=0)
    concurrency: int = Field(PERFORMANCE_CONCURRENCY, gt=0)
    requests: int = Field(PERFORMANCE_REQUESTS, gt=0)


class Evaluate(Strict):
    """Benchmarks from the catalog, run against a model or a running Endpoint."""

    # Validation names the Pipeline's last Model Version, `@quantize` or `@finetune`, when none is
    # given.
    model: (
        BaseModelReference
        | ModelReference
        | EndpointReference
        | Literal[FINETUNE_OUTPUT, QUANTIZE_OUTPUT]
        | None
    ) = None
    benchmarks: list[Literal[tuple(BENCHMARKS)]] = []
    # Samples per task, for a quick look; scores on fewer samples don't compare with full runs.
    limit: int | None = Field(None, gt=0)
    # For the vLLM `evaluate` starts; an Endpoint serves with its own.
    serving: ServingOptions | None = None
    # Measured only when set, so evaluations stay fast by default.
    performance: Performance | None = None

    @model_validator(mode="after")
    def check_something_to_run(self) -> "Evaluate":
        if not (self.benchmarks or self.performance):
            raise ValueError("name `benchmarks`, `performance` or both")
        return self


class Serve(ServingOptions):
    """An Endpoint for the Pipeline's last Model Version, started once the Pipeline made it."""

    # Validation pins the Pipeline's name when none is given.
    name: EndpointName | None = Field(None, title="Endpoint name")


class PipelineRequest(Strict):
    schema_version: Literal[1] = 1
    name: str = Field(pattern=f"^{MODEL_NAME_PATTERN}$", max_length=63)
    distill: Distill | None = None
    finetune: Finetune | None = None
    quantize: Quantize | None = None
    evaluate: Evaluate | None = None
    serve: Serve | None = None

    @model_validator(mode="after")
    def check_stages(self) -> "PipelineRequest":
        if not (self.distill or self.finetune or self.quantize or self.evaluate):
            raise ValueError(
                "enable at least one of `distill`, `finetune`, `quantize` and `evaluate`"
            )
        if self.serve and not (self.finetune or self.quantize):
            raise ValueError("`serve` serves the output of `finetune` or `quantize`; enable either")
        return self
