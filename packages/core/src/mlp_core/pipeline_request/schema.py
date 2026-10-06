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
    ADAPTER_METHODS,
    ALGORITHMS,
    ASSISTANT_ONLY_LOSS_INFOBOX,
    BACKENDS,
    BENCHMARKS,
    CALIBRATION_DATASET,
    CALIBRATION_MAX_LENGTH,
    CALIBRATION_SAMPLES,
    DISTILL_MAX_TOKENS,
    DISTILL_OUTPUT,
    DISTILL_TEMPERATURE,
    FINETUNE_OUTPUT,
    PERFORMANCE_CONCURRENCY,
    PERFORMANCE_OUTPUT_TOKENS,
    PERFORMANCE_PROMPT_TOKENS,
    PERFORMANCE_REQUESTS,
    QUANTIZATION_SCHEME,
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
    SWEEP_TRIALS,
    UNCALIBRATED_SCHEMES,
    WEIGHT_METHODS,
)
from mlp_core.endpoint_spec import ENDPOINT_NAME_PATTERN, ServingOptions
from mlp_core.pipeline_request.references import (
    MODEL_NAME_PATTERN,
    BaseModelReference,
    DatasetReference,
    ModelReference,
)

EndpointReference = Annotated[str, Field(pattern=f"^endpoint:{ENDPOINT_NAME_PATTERN}$")]
InClusterTeacher = TypeAdapter(BaseModelReference | ModelReference | EndpointReference)
JobTeacher = TypeAdapter(BaseModelReference | ModelReference)


# Clients show a field only where its trait holds, e.g. a Teacher only for an algorithm that learns
# from one; GET /schema publishes what each trait depends on.
def applies_if(trait: str, **extra) -> dict:
    return {"json_schema_extra": {"applies_if": trait, **extra}}


# A field's docstring is its `description` in the published schema, which clients show as help.
class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)


class TrainerSettings(BaseModel):
    """Requires the basic values and allows every other field of its TRL/PEFT config class."""

    model_config = ConfigDict(extra="allow", use_attribute_docstrings=True)


# Checked against the TRL config of the Phase's algorithm, which also requires its length setting;
# the algorithm's defaults fill in what it leaves out.
class PhaseSettings(TrainerSettings):
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


# Checked against PEFT's LoraConfig; the algorithm's defaults fill in what it leaves out.
class LoraSettings(TrainerSettings):
    r: int = Field(gt=0)
    """The Adapter's rank: higher learns more and takes more memory."""
    lora_alpha: int
    """Scales the Adapter's update; often twice the rank."""
    lora_dropout: float
    """The share of the Adapter's inputs dropped while training, against overfitting."""
    target_modules: str | list[str]
    """The layers that get an Adapter; `all-linear` means every linear layer."""


class Reward(Strict):
    weight: float
    """How much this reward counts in the sum the model learns from."""
    # Runs only in the Sandbox (#19).
    source: str = Field(max_length=REWARD_SOURCE_MAX_LENGTH, json_schema_extra={"format": "python"})
    """Python that defines `def reward(sample, item)`, returning a float or None."""


# Names its rewards/<name>/… metrics.
RewardName = Annotated[str, Field(pattern=r"^[\w-]{1,64}$")]


class PhaseConfiguration(Strict):
    """What one training run needs: a Phase's, or each Trial's of a Sweep."""

    algorithm: Literal[tuple(ALGORITHMS)]
    """How the model learns: `sft` imitates examples, `dpo` and `kto` learn preferences,
    `distillation` a Teacher's token probabilities, `grpo` and `rloo` from rewards."""
    dataset: DatasetReference | Literal[DISTILL_OUTPUT]
    """The Dataset it trains on, in a row format the algorithm reads."""
    method: Literal[WEIGHT_METHODS] = "lora"
    """`lora` trains a small Adapter, `qlora` one on a 4-bit base to save memory, `full` every
    weight."""
    # Clients show the algorithm's published `settings` for it.
    settings: PhaseSettings = Field(
        default_factory=dict, json_schema_extra={"trainer_settings": "settings"}
    )
    """The algorithm's TRL settings, over its defaults."""
    # After a kept Adapter, a Phase continues it as it is (#19); a new one takes the defaults.
    lora: LoraSettings | None = Field(
        None, **applies_if("new_adapter", trainer_settings="lora_settings")
    )
    """A new Adapter's settings, from PEFT's LoraConfig, over the algorithm's defaults."""
    teacher: str | None = Field(
        None, **applies_if("learns_from_teacher", references=["hf:", "model:"])
    )
    """The Base Model or Model Version whose token probabilities a `distillation` Phase learns."""
    rewards: dict[RewardName, Reward] | None = Field(
        None, description=REWARDS_INFOBOX, **applies_if("learns_from_rewards")
    )

    @model_validator(mode="before")
    @classmethod
    def with_default_settings(cls, data):
        """The data with its settings, and its Adapter's if it names any, over the defaults."""
        algorithm = ALGORITHMS.get(data.get("algorithm")) if isinstance(data, dict) else None
        if algorithm is None:
            return data
        data = {
            **data,
            "settings": {**algorithm["default_settings"], **(data.get("settings") or {})},
        }
        if isinstance(data.get("lora"), dict):
            data["lora"] = {**algorithm["default_lora"], **data["lora"]}
        return data

    def with_default_lora(self, continues_adapter: bool) -> None:
        """Gives a new Adapter its algorithm's LoRA defaults, unless the Phase names its own."""
        if self.method in ADAPTER_METHODS and not continues_adapter and self.lora is None:
            self.lora = LoraSettings(**ALGORITHMS[self.algorithm]["default_lora"])

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
    output: Literal["adapter", "merged"] = Field("adapter", **applies_if("trains_adapter"))
    """`adapter` registers the Adapter; `merged` the Adapter merged into its base, as full weights
    that stand alone."""
    params_from: Literal[SWEEP_OUTPUT] | None = None
    """`@sweep` trains with the best parameters the `sweep` Stage found, over the Phase's own."""

    @property
    def keeps_adapter(self) -> bool:
        return self.method in ADAPTER_METHODS and self.output == "adapter"

    @property
    def merges_adapter(self) -> bool:
        return self.method in ADAPTER_METHODS and self.output == "merged"

    def with_parameters(self, parameters: dict) -> "Phase":
        """The Phase with `{settings: …, lora: …}` values over its own."""
        settings = {**self.settings.model_dump(), **parameters.get("settings", {})}
        lora = self.lora and {**self.lora.model_dump(), **parameters.get("lora", {})}
        return Phase.model_validate({**self.model_dump(), "settings": settings, "lora": lora})


class SweepParameter(Strict):
    """A range to sample from, `{min, max, scale}`, or the `values` to choose between."""

    min: int | float | None = None
    """The smallest value to try; with an integer `max` too, only integers are tried."""
    max: int | float | None = None
    """The largest value to try."""
    scale: Literal["linear", "log"] = "linear"
    """`log` samples evenly across orders of magnitude, e.g. for learning rates."""
    values: list[bool | int | float | str] | None = Field(None, min_length=1)
    """The values to choose between, instead of a range."""

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
    """TRL settings the Trials vary, by name."""
    lora: dict[str, SweepParameter] = {}
    """LoraConfig settings the Trials vary, by name."""


class Objective(Strict):
    # From the catalog's list for the algorithm.
    metric: str
    """The metric each Trial is judged by, one the algorithm logs."""
    goal: Literal["minimize", "maximize"]
    """Whether a lower or a higher `metric` is better."""


class Sweep(PhaseConfiguration):
    """A search for the parameters of one Phase configuration with Optuna, Trial after Trial."""

    # Validation names `finetune`'s starting model when none is given.
    model: BaseModelReference | ModelReference | None = None
    """The model each Trial starts from; `finetune`'s starting model when left out."""
    backend: Literal[tuple(BACKENDS)] = "hf"
    """`hf` trains with TRL and PEFT; `unsloth` with Unsloth, faster, on one GPU."""
    parameters: SweepParameters
    """The settings the Trials vary, and their ranges."""
    # Validation sets the algorithm's default, which clients find as its published `objective`.
    objective: Objective | None = Field(None, json_schema_extra={"algorithm_defaults": "objective"})
    """What makes one Trial better than another; the algorithm's first metric when left out."""
    trials: int = Field(SWEEP_TRIALS, gt=0)
    """How many training runs to try; their weights are thrown away."""
    sampler: Literal[SWEEP_SAMPLERS] = "tpe"
    """`tpe` learns from earlier Trials, `random` samples blindly, `grid` tries every combination
    of the parameters' `values`, at most `trials` of them."""
    # Validation sets the default.
    eval_split: float | None = Field(None, gt=0, lt=1)
    """The share of the Dataset's rows held out to measure the objective on."""
    eval_dataset: DatasetReference | None = None
    """A Dataset to measure the objective on, instead of held-out rows."""

    @model_validator(mode="after")
    def check_search(self) -> "Sweep":
        parameters = {**self.parameters.settings, **self.parameters.lora}
        if not parameters:
            raise ValueError("name at least one of `parameters.settings` or `parameters.lora`")
        if self.sampler == "grid" and any(p.values is None for p in parameters.values()):
            raise ValueError("`grid` tries listed values; give every parameter `values`")
        objectives, goal = (
            ALGORITHMS[self.algorithm]["objectives"],
            ALGORITHMS[self.algorithm]["goal"],
        )
        self.objective = self.objective or Objective(metric=objectives[0], goal=goal)
        if self.objective.metric not in objectives:
            raise ValueError(
                f"a {self.algorithm} Sweep optimizes one of {', '.join(objectives)}, not "
                f"{self.objective.metric}"
            )
        if self.eval_split is not None and self.eval_dataset is not None:
            raise ValueError("name `eval_split` or `eval_dataset`, not both")
        if self.eval_dataset is None and self.eval_split is None:
            self.eval_split = SWEEP_EVAL_SPLIT
        # Its Trials train like a first Phase.
        self.with_default_lora(continues_adapter=False)
        return self

    def trial_phase(self, parameters: dict) -> Phase:
        """The Phase a Trial trains with `{settings: …, lora: …}`; it is never saved or served."""
        fields = self.model_dump(include=set(PhaseConfiguration.model_fields))
        # Its Adapter is thrown away, so it needn't be one vLLM serves.
        return Phase.model_validate({**fields, "output": "merged"}).with_parameters(parameters)


class Finetune(Strict):
    # `from` is a Python keyword, so the field has another name and is read and written as `from`.
    model_config = ConfigDict(
        extra="forbid", serialize_by_alias=True, use_attribute_docstrings=True
    )

    base_model: BaseModelReference | None = None
    """The open-source model the first Phase starts from."""
    from_: ModelReference | None = Field(None, alias="from", title="From Model Version")
    """A full-weight Model Version the first Phase starts from, instead of a Base Model."""
    backend: Literal[tuple(BACKENDS)] = "hf"
    """`hf` trains with TRL and PEFT; `unsloth` with Unsloth, faster, on one GPU."""
    phases: list[Phase] = Field(min_length=1)
    """Run in order, each starting from the Model Version the one before it registered."""

    @model_validator(mode="after")
    def check_one_starting_model(self) -> "Finetune":
        if (self.base_model is None) == (self.from_ is None):
            raise ValueError("name exactly one of `base_model` and `from`")
        return self

    @model_validator(mode="after")
    def with_default_loras(self) -> "Finetune":
        for index, phase in enumerate(self.phases):
            phase.with_default_lora(index > 0 and self.phases[index - 1].keeps_adapter)
        return self

    @property
    def starting_model(self) -> str:
        return self.base_model or self.from_


class ToolFunction(Strict):
    name: str = Field(pattern=r"^[\w-]{1,64}$")
    """The name the Teacher calls the tool by."""
    description: str | None = None
    """What the tool does, for the Teacher."""
    # Validation checks that it is a JSON Schema.
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    """The JSON Schema of the tool's arguments."""


class Tool(Strict):
    """A function the Teacher may call, in the OpenAI `tools` format."""

    type: Literal["function"] = "function"
    """Always `function`."""
    function: ToolFunction
    """The function the Teacher may call."""


class Distill(Strict):
    """A Distillation Dataset: each prompt's single Teacher reply, a text or tool calls (#17)."""

    dataset: DatasetReference
    """The prompts to ask the Teacher: a Dataset of `prompt` rows."""
    # Any other name is a model at `api_url`.
    teacher: str = Field(json_schema_extra={"references": ["hf:", "model:", "endpoint:"]})
    """A Base Model or Model Version run here, a running Endpoint, or the name of a model at
    `api_url`."""
    api_url: str | None = Field(
        None, pattern=r"^https?://\S+$", title="API URL", **applies_if("external_teacher")
    )
    """An OpenAI-compatible API serving the Teacher, e.g. https://api.openai.com/v1; its key is
    the Teacher API key Secret."""
    tools: list[Tool] | None = None
    """Tools offered with every prompt; a reply may call them."""
    parallel_tool_calls: bool = False
    """Off: the Teacher is asked for one call per reply, and replies with several are dropped."""
    max_tokens: int | None = Field(DISTILL_MAX_TOKENS, gt=0)
    """The longest reply the Teacher may write, in tokens; a cut-off reply is dropped."""
    temperature: float | None = Field(DISTILL_TEMPERATURE, ge=0)
    """How varied the Teacher's replies are: 0 always picks the likeliest token."""
    serving: ServingOptions | None = None
    """How vLLM serves a Base Model or Model Version Teacher."""

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

    # The install registers the default one.
    dataset: DatasetReference = f"dataset:{CALIBRATION_DATASET}"
    """A Dataset of messages or text rows like the traffic the model will see."""
    samples: int = Field(CALIBRATION_SAMPLES, gt=0)
    """How many rows to measure on."""
    max_length: int = Field(CALIBRATION_MAX_LENGTH, gt=0)
    """Tokens per row; longer rows are cut."""


class Quantize(Strict):
    """A quantized copy of a model by llm-compressor, registered as full weights."""

    # Validation names `@finetune` when none is given.
    model: BaseModelReference | ModelReference | Literal[FINETUNE_OUTPUT] | None = None
    """The model to quantize, `@finetune`'s output when left out; an Adapter is merged first."""
    scheme: Literal[tuple(QUANTIZATION_SCHEMES)] = QUANTIZATION_SCHEME
    """`fp8-dynamic` needs no data and suits recent GPUs; the `w4a16` ones make 4-bit weights,
    the smallest; `w8a8-int8` makes 8-bit weights and activations."""
    ignore: list[str] = QUANTIZE_IGNORE
    """Layers kept unquantized."""
    calibration: Calibration | None = None
    """The rows a calibrated scheme measures activations on."""

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
    # Their hidden states fill the step's disk.
    samples: int = Field(SPECULATE_SAMPLES, gt=0)
    """Conversations rendered through the verifier to learn from."""
    seq_length: int = Field(SPECULATE_SEQ_LENGTH, gt=0)
    """Tokens per conversation, and the length batches are packed to."""
    epochs: int = Field(SPECULATE_EPOCHS, gt=0)
    """How many times training goes through the conversations."""
    learning_rate: float = Field(SPECULATE_LEARNING_RATE, gt=0)
    """How far each optimizer step moves the Speculator's weights."""
    draft_vocab_size: int = Field(SPECULATE_DRAFT_VOCAB_SIZE, gt=0)
    """The tokens the Speculator may propose, the most frequent ones."""


class Speculate(Strict):
    """A Speculator trained with `speculators` on the verifier's hidden states, for it alone."""

    speculator: Literal[SPECULATORS] = "eagle3"
    """The Speculator type; each is served with its own speculative method."""
    # Validation names `@quantize`, else `@finetune`, when none is given.
    model: (
        BaseModelReference | ModelReference | Literal[FINETUNE_OUTPUT, QUANTIZE_OUTPUT] | None
    ) = None
    """The model it drafts for, as full weights; the Pipeline's last one when left out."""
    dataset: DatasetReference | Literal[DISTILL_OUTPUT]
    """Conversations like the traffic the Endpoint will see; `@distill`'s replies count."""
    settings: SpeculateSettings = SpeculateSettings()
    """How the Speculator trains."""


class Performance(Strict):
    """Serving performance by GuideLLM: synthetic chat requests, a fixed number at once."""

    prompt_tokens: int = Field(PERFORMANCE_PROMPT_TOKENS, gt=0)
    """Tokens in each request's prompt."""
    output_tokens: int = Field(PERFORMANCE_OUTPUT_TOKENS, gt=0)
    """Tokens each request asks for."""
    concurrency: int = Field(PERFORMANCE_CONCURRENCY, gt=0)
    """Requests sent at once."""
    requests: int = Field(PERFORMANCE_REQUESTS, gt=0)
    """Requests in all."""


class Evaluate(Strict):
    """Benchmarks from the catalog, run against a model or a running Endpoint."""

    # Validation names `@quantize` or `@finetune` when none is given.
    model: (
        BaseModelReference
        | ModelReference
        | EndpointReference
        | Literal[FINETUNE_OUTPUT, QUANTIZE_OUTPUT]
        | None
    ) = None
    """The model to evaluate; the Pipeline's last Model Version when left out."""
    benchmarks: list[Literal[tuple(BENCHMARKS)]] = []
    """Benchmarks from the catalog."""
    limit: int | None = Field(None, gt=0)
    """Samples per benchmark, for a quick look; such scores don't compare with full runs."""
    serving: ServingOptions | None = None
    """How vLLM serves the model; an Endpoint serves with its own."""
    # Measured only when set, so evaluations stay fast by default.
    performance: Performance | None = None
    """Serving speed, measured with synthetic chat requests."""

    @model_validator(mode="after")
    def check_something_to_run(self) -> "Evaluate":
        if not (self.benchmarks or self.performance):
            raise ValueError("name `benchmarks`, `performance` or both")
        return self


class PipelineRequest(Strict):
    schema_version: Literal[1] = 1
    """The version of this schema."""
    name: str = Field(pattern=f"^{MODEL_NAME_PATTERN}$", max_length=63)
    """Names the Pipeline, and the Registered Model and Dataset it makes."""
    distill: Distill | None = None
    """Asks a Teacher about prompts and keeps its replies as a Distillation Dataset."""
    sweep: Sweep | None = None
    """Searches the best settings for one Phase, Trial after Trial."""
    finetune: Finetune | None = None
    """Trains Phases one after another, each registering a Model Version."""
    quantize: Quantize | None = None
    """Makes a smaller, faster copy of a model with fewer bits per weight."""
    speculate: Speculate | None = None
    """Trains a Speculator, a small draft model that speeds up serving one model."""
    evaluate: Evaluate | None = None
    """Scores a model on benchmarks, and optionally its serving speed."""

    @model_validator(mode="after")
    def check_stages(self) -> "PipelineRequest":
        stages = (self.distill, self.sweep, self.finetune, self.quantize, self.speculate)
        if not any((*stages, self.evaluate)):
            raise ValueError(
                "enable at least one of `distill`, `sweep`, `finetune`, `quantize`, `speculate` "
                "and `evaluate`"
            )
        return self
