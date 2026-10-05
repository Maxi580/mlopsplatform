import json
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlp_core.config import (
    ENDPOINT_NAME_MAX_LENGTH,
    SPECULATIVE_METHODS,
    SPECULATIVE_TOKENS,
    VLLM_LORA_RANKS,
    VLLM_TOOL_PARSERS,
)
from mlp_core.pipeline_request.references import BaseModelReference, ModelReference

# Names the Endpoint's Kubernetes objects and its URL, so it must be a DNS label.
ENDPOINT_NAME_PATTERN = r"[a-z0-9]([a-z0-9-]*[a-z0-9])?"
EndpointName = Annotated[
    str, Field(pattern=f"^{ENDPOINT_NAME_PATTERN}$", max_length=ENDPOINT_NAME_MAX_LENGTH)
]


class ServingOptions(BaseModel):
    """The curated serving options; GPUs come from platform settings, never from here."""

    model_config = ConfigDict(extra="forbid")

    max_model_len: int | None = Field(None, gt=0)
    prefix_caching: bool = True
    dtype: Literal["auto", "half", "float16", "bfloat16", "float32"] | None = None
    gpu_memory_utilization: float | None = Field(None, gt=0, le=1)
    max_num_seqs: int | None = Field(None, gt=0)
    max_num_batched_tokens: int | None = Field(None, gt=0)
    async_scheduling: bool | None = None
    kv_cache_dtype: Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2"] | None = None
    # On the fly, at load time.
    quantization: Literal["fp8", "bitsandbytes"] | None = None
    # Overrides the parser the model's `model_type` maps to.
    tool_parser: Literal[VLLM_TOOL_PARSERS] | None = None


class Speculative(BaseModel):
    """Speculative decoding: `ngram` looks the next tokens up in the context, the others draft."""

    model_config = ConfigDict(extra="forbid")

    method: Literal[tuple(SPECULATIVE_METHODS)]
    # A Speculator trained for the served model, or a small model for `draft`.
    model: BaseModelReference | ModelReference | None = None
    num_speculative_tokens: int = Field(SPECULATIVE_TOKENS, gt=0)
    prompt_lookup_min: int | None = Field(None, gt=0)
    prompt_lookup_max: int | None = Field(None, gt=0)

    @model_validator(mode="after")
    def check_method(self) -> "Speculative":
        ngram = self.method == "ngram"
        if ngram and self.model:
            raise ValueError("`ngram` drafts from the context; leave out `model`")
        if not ngram and not self.model:
            raise ValueError(f"`{self.method}` drafts with a model; name it in `model`")
        if not ngram and (self.prompt_lookup_min or self.prompt_lookup_max):
            raise ValueError("`prompt_lookup_min` and `prompt_lookup_max` are for `ngram` only")
        return self


class EndpointSpec(ServingOptions):
    model: BaseModelReference | ModelReference
    speculative: Speculative | None = None


@dataclass(frozen=True)
class VllmModel:
    """Where vLLM finds the model: a directory, or a Hugging Face repo at a cached revision."""

    path: str
    revision: str | None = None
    adapter_path: str | None = None
    adapter_rank: int | None = None
    # The parser the model's `model_type` maps to; None serves it without tool calling.
    tool_parser: str | None = None
    # Where vLLM finds the speculative `model`, if one drafts.
    drafter_path: str | None = None
    drafter_revision: str | None = None


# Options whose vLLM flag has another name than the option.
VLLM_FLAGS = {"prefix_caching": "enable-prefix-caching"}


def vllm_args(
    spec: ServingOptions,
    model: VllmModel,
    served_name: str,
    gpus: int,
    speculative: Speculative | None = None,
) -> list[str]:
    """The `vllm serve` arguments for the spec, shared by Endpoints, `evaluate` and Teachers."""
    # 1. The model, which clients call `served_name`; an Adapter takes the name from its base.
    base_name = f"{served_name}-base" if model.adapter_path else served_name
    args = [model.path, "--served-model-name", base_name, "--tensor-parallel-size", str(gpus)]
    # `evaluate` runs lm-eval against vLLM, and it tokenizes through vLLM's tokenizer endpoints.
    args.append("--enable-tokenizer-info-endpoint")
    if model.revision:
        args += ["--revision", model.revision, "--tokenizer-revision", model.revision]
    if model.adapter_path:
        rank = min(r for r in VLLM_LORA_RANKS if r >= model.adapter_rank)
        lora = f"{served_name}={model.adapter_path}"
        args += ["--enable-lora", "--lora-modules", lora, "--max-lora-rank", str(rank)]

    # 2. The serving options that are set; a boolean one is switched on or off explicitly.
    options = spec.model_dump(exclude={"model", "tool_parser", "speculative"}, exclude_none=True)
    for option, value in options.items():
        flag = VLLM_FLAGS.get(option, option.replace("_", "-"))
        if isinstance(value, bool):
            args.append(f"--{flag}" if value else f"--no-{flag}")
        else:
            args += [f"--{flag}", str(value)]

    # 3. Tool calling, whenever a parser is known (#16).
    tool_parser = spec.tool_parser or model.tool_parser
    if tool_parser:
        args += ["--enable-auto-tool-choice", "--tool-call-parser", tool_parser]

    # 4. Speculative decoding, an Endpoint option only.
    if speculative:
        args += ["--speculative-config", json.dumps(speculative_config(speculative, model))]
    return args


def speculative_config(speculative: Speculative, model: VllmModel) -> dict:
    """vLLM's speculative config for the method, drafting with the model vLLM finds."""
    settings = speculative.model_dump(exclude={"method", "model"}, exclude_none=True)
    drafter = {"model": model.drafter_path, "revision": model.drafter_revision}
    drafter = {key: value for key, value in drafter.items() if value}
    return {**SPECULATIVE_METHODS[speculative.method], **drafter, **settings}
