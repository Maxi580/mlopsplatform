from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from mlp_core.config import ENDPOINT_NAME_MAX_LENGTH, VLLM_LORA_RANKS, VLLM_TOOL_PARSERS
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


class EndpointSpec(ServingOptions):
    model: BaseModelReference | ModelReference


@dataclass(frozen=True)
class VllmModel:
    """Where vLLM finds the model: a directory, or a Hugging Face repo at a cached revision."""

    path: str
    revision: str | None = None
    adapter_path: str | None = None
    adapter_rank: int | None = None
    # The parser the model's `model_type` maps to; None serves it without tool calling.
    tool_parser: str | None = None


# Options whose vLLM flag has another name than the option.
VLLM_FLAGS = {"prefix_caching": "enable-prefix-caching"}


def vllm_args(spec: EndpointSpec, model: VllmModel, served_name: str, gpus: int) -> list[str]:
    """The `vllm serve` arguments for the spec, shared by Endpoints, `evaluate` and Teachers."""
    # 1. The model, which clients call `served_name`; an Adapter takes the name from its base.
    base_name = f"{served_name}-base" if model.adapter_path else served_name
    args = [model.path, "--served-model-name", base_name, "--tensor-parallel-size", str(gpus)]
    if model.revision:
        args += ["--revision", model.revision, "--tokenizer-revision", model.revision]
    if model.adapter_path:
        rank = min(r for r in VLLM_LORA_RANKS if r >= model.adapter_rank)
        lora = f"{served_name}={model.adapter_path}"
        args += ["--enable-lora", "--lora-modules", lora, "--max-lora-rank", str(rank)]

    # 2. The serving options that are set; a boolean one is switched on or off explicitly.
    options = spec.model_dump(exclude={"model", "tool_parser"}, exclude_none=True)
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
    return args
