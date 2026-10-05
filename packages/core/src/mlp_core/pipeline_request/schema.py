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
    SPECULATE_DRAFT_VOCAB_SIZE,
    SPECULATE_EPOCHS,
    SPECULATE_LEARNING_RATE,
    SPECULATE_SAMPLES,
    SPECULATE_SEQ_LENGTH,
    SPECULATORS,
    SWEEP_EVAL_SPLIT,
    SWEEP_OUTPUT,
    SWEEP_SAMPLERS,
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


class PhaseConfiguration(Strict):
    """What one training run needs: a Phase's, or each Trial's of a Sweep."""

    algorithm: Literal[tuple(ALGORITHMS)]
    dataset: DatasetReference | Literal[DISTILL_OUTPUT]
    method: Literal[WEIGHT_METHODS] = "lora"
    settings: PhaseSettings
    # A new Adapter's settings; after a kept Adapter, a Phase continues it as it is (#19).
    lora: LoraSettings | None = None
    # The Base Model or Model Version a `distillation` Phase learns the token probabilities of.
    teacher: str | None = None
    rewards: dict[RewardName, Reward] | None = Field(None, description=REWARDS_INFOBOX)

    @model_validator(mode="after")
    def check_rewards(self) -> "PhaseConfiguration":
        if not ALGORITHMS[self.algorithm]["learns_from_rewards"]:
            if self.rewards is not None:
                raise ValueError(f"a {self.algorithm} Phase has no `rewards`")
        elif not self.rewards:
            raise ValueError(f"a {self.algorithm} Phase learns from `rewards`; name at least one")
        return self

    @model_validator(mode="after")
    def check_teacher(self) -> "PhaseConfiguration":
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


class Phase(PhaseConfiguration):
    # `merged` registers the Adapter merged into its base, as full weights; `full` ignores it.
    output: Literal["adapter", "merged"] = "adapter"
    # The `sweep` Stage's best parameters, over the Phase's own `settings` and `lora`.
    params_from: Literal[SWEEP_OUTPUT] | None = None

    @property
    def keeps_adapter(self) -> bool:
        return self.method != "full" and self.output == "adapter"

    @property
    def merges_adapter(self) -> bool:
        return self.method != "full" and self.output == "merged"

    def with_parameters(self, parameters: dict) -> "Phase":
        """The Phase with `{settings: …, lora: …}` values over its own."""
        settings = {**self.settings.model_dump(), **parameters.get("settings", {})}
        lora = self.lora and {**self.lora.model_dump(), **parameters.get("lora", {})}
        return Phase.model_validate({**self.model_dump(), "settings": settings, "lora": lora})


class SweepParameter(Strict):
    """A range to sample from, `{min, max, scale}`, or the `values` to choose between."""

    # Both integers sample integers.
    min: int | float | None = None
    max: int | float | None = None
    scale: Literal["linear", "log"] = "linear"
    values: list[bool | int | float | str] | None = Field(None, min_length=1)

    @model_validator(mode="after")
    def check_range_or_values(self) -> "SweepParameter":
        is_range = self.min is not None or self.max is not None
        if is_range == (self.values is not None):
            raise ValueError("give either `min` and `max`, or `values`")
        if is_range and (self.min is None or self.max is None or self.min >= self.max):
            raise ValueError("give a `min` below its `max`")
        if is_range and self.scale == "log" and self.min <= 0:
            raise ValueError("a `log` scale needs a `min` above 0")
        return self


class SweepParameters(Strict):
    """The settings Trials vary, by name: TRL config fields, and LoraConfig fields."""

    settings: dict[str, SweepParameter] = {}
    lora: dict[str, SweepParameter] = {}


class Objective(Strict):
    # From the catalog's list for the algorithm.
    metric: str
    goal: Literal["minimize", "maximize"]


class Sweep(PhaseConfiguration):
    """A search for the parameters of one Phase configuration with Optuna, Trial after Trial."""

    # Validation names `finetune`'s starting model when none is given.
    model: BaseModelReference | ModelReference | None = None
    backend: Literal[tuple(BACKENDS)] = "hf"
    parameters: SweepParameters
    objective: Objective
    trials: int = Field(gt=0)
    # `grid` tries every combination of the parameters' `values`, at most `trials` of them.
    sampler: Literal[SWEEP_SAMPLERS] = "tpe"
    # The share of the Dataset's rows held out to measure the objective on, unless `eval_dataset`
    # is named; validation sets the default.
    eval_split: float | None = Field(None, gt=0, lt=1)
    eval_dataset: DatasetReference | None = None

    @model_validator(mode="after")
    def check_search(self) -> "Sweep":
        parameters = {**self.parameters.settings, **self.parameters.lora}
        if not parameters:
            raise ValueError("name at least one of `parameters.settings` or `parameters.lora`")
        if self.sampler == "grid" and any(p.values is None for p in parameters.values()):
            raise ValueError("`grid` tries listed values; give every parameter `values`")
        objectives = ALGORITHMS[self.algorithm]["objectives"]
        if self.objective.metric not in objectives:
            raise ValueError(
                f"a {self.algorithm} Sweep optimizes one of {', '.join(objectives)}, not "
                f"{self.objective.metric}"
            )
        if self.eval_split is not None and self.eval_dataset is not None:
            raise ValueError("name `eval_split` or `eval_dataset`, not both")
        if self.eval_dataset is None and self.eval_split is None:
            self.eval_split = SWEEP_EVAL_SPLIT
        return self

    def trial_phase(self, parameters: dict) -> Phase:
        """The Phase a Trial trains with `{settings: …, lora: …}`; it is never saved or served."""
        fields = self.model_dump(include=set(PhaseConfiguration.model_fields))
        # Its Adapter is thrown away, so it needn't be one vLLM serves.
        return Phase.model_validate({**fields, "output": "merged"}).with_parameters(parameters)


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


class SpeculateSettings(Strict):
    # Conversations rendered through the verifier; their hidden states fill the step's disk.
    samples: int = Field(SPECULATE_SAMPLES, gt=0)
    # Tokens per conversation, and the length batches are packed to.
    seq_length: int = Field(SPECULATE_SEQ_LENGTH, gt=0)
    epochs: int = Field(SPECULATE_EPOCHS, gt=0)
    learning_rate: float = Field(SPECULATE_LEARNING_RATE, gt=0)
    # The tokens the Speculator may propose, picked by frequency.
    draft_vocab_size: int = Field(SPECULATE_DRAFT_VOCAB_SIZE, gt=0)


class Speculate(Strict):
    """A Speculator trained with `speculators` on the verifier's hidden states, for it alone."""

    speculator: Literal[SPECULATORS] = "eagle3"
    # The verifier; validation names `@quantize`, else `@finetune`, when none is given.
    model: (
        BaseModelReference | ModelReference | Literal[FINETUNE_OUTPUT, QUANTIZE_OUTPUT] | None
    ) = None
    # Conversations like the traffic the Endpoint will see; `@distill`'s replies count as ones.
    dataset: DatasetReference | Literal[DISTILL_OUTPUT]
    settings: SpeculateSettings = SpeculateSettings()


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
    sweep: Sweep | None = None
    finetune: Finetune | None = None
    quantize: Quantize | None = None
    speculate: Speculate | None = None
    evaluate: Evaluate | None = None
    serve: Serve | None = None

    @model_validator(mode="after")
    def check_stages(self) -> "PipelineRequest":
        stages = (self.distill, self.sweep, self.finetune, self.quantize, self.speculate)
        if not any((*stages, self.evaluate)):
            raise ValueError(
                "enable at least one of `distill`, `sweep`, `finetune`, `quantize`, `speculate` "
                "and `evaluate`"
            )
        if self.serve and not (self.finetune or self.quantize):
            raise ValueError("`serve` serves the output of `finetune` or `quantize`; enable either")
        return self
