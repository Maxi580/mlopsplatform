from dataclasses import dataclass

from mlp_core import config
from mlp_core.endpoint_spec import ServingOptions, serving_share
from mlp_core.pipeline_request.schema import PhaseConfiguration
from mlp_core.settings import Settings


@dataclass(frozen=True)
class Gpu:
    memory_bytes: int
    shared_with_system: bool


@dataclass(frozen=True)
class ModelMemory:
    weights_bytes: int
    # 2 × layers × KV heads × head dimension: the values the KV cache holds per token.
    kv_values_per_token: int
    dtype_bytes: int
    # The longest prompt plus reply the model allows, if its config says.
    max_length: int | None


def with_gpu_share(
    options: ServingOptions,
    model: ModelMemory | None,
    gpu: Gpu | None,
    gpus: int,
    settings: Settings,
) -> ServingOptions:
    """The options with a GPU Share: the user's, else calculated, else the fixed one."""
    # 1. A model whose files don't say keeps the fixed share and vLLM's own length.
    update = {"gpu_memory_utilization": serving_share(options)}
    if model is None:
        return options.model_copy(update=update)

    # 2. The length is always passed, as the KV cache is sized for it.
    longest = settings.gpu_share_max_model_len
    length = options.max_model_len or min(model.max_length or longest, longest)
    update["max_model_len"] = length

    # 3. A value the user entered wins, and an unknown GPU has no share to work out.
    if options.gpu_memory_utilization is not None or gpu is None:
        return options.model_copy(update=update)

    # 4. The KV cache for several requests at once, if that fits.
    def share_with_kv_cache(tokens: int, dtype_bytes: int) -> float:
        reserve = settings.gpu_share_vllm_reserve_gib
        if gpu.shared_with_system:
            reserve += settings.gpu_share_shared_memory_reserve_gib
        per_gpu = settings.gpu_share_weights_factor * model.weights_bytes
        per_gpu += model.kv_values_per_token * dtype_bytes * tokens
        reserve_bytes = reserve * config.QUANTITY_SUFFIXES["Gi"]
        return settings.share_in_slices((per_gpu / gpus + reserve_bytes) / gpu.memory_bytes)

    largest = largest_calculated_share(gpu, settings)
    dtype_bytes = kv_cache_dtype_bytes(options, model)
    share = share_with_kv_cache(max(length, settings.gpu_share_kv_cache_tokens), dtype_bytes)

    # 5. Else for one request of the longest length.
    if share > largest:
        share = share_with_kv_cache(length, dtype_bytes)

    # 6. Else in `fp8`, up to the maximum, where the job runs and says so if it can't.
    if share > largest:
        if options.kv_cache_dtype in (None, "auto"):
            update["kv_cache_dtype"] = "fp8"
        share = min(share_with_kv_cache(length, 1), largest)
    return options.model_copy(update={**update, "gpu_memory_utilization": share})


def training_gpu_share(
    weights_bytes: int,
    method: str | None,
    teacher_bytes: int,
    colocated_share: float,
    gpu: Gpu,
    settings: Settings,
) -> float:
    """The GPU Share to train the weights: a Teacher's sit beside them, colocated vLLM on top."""
    factor = settings.gpu_share_training_factor
    if method == "full":
        factor = settings.gpu_share_full_training_factor
    reserve = settings.gpu_share_training_reserve_gib * config.QUANTITY_SUFFIXES["Gi"]
    need = factor * weights_bytes + teacher_bytes + reserve
    return min(
        settings.share_in_slices(need / gpu.memory_bytes + colocated_share),
        largest_calculated_share(gpu, settings),
    )


def model_memory(model_config: dict, weights_bytes: int) -> ModelMemory | None:
    """What the model takes on a GPU, from its config.json and weights; None if it doesn't say."""
    # Multimodal models nest their language model's config.
    text = model_config.get("text_config") or model_config
    try:
        layers, heads = text["num_hidden_layers"], text["num_attention_heads"]
        kv_heads = text.get("num_key_value_heads") or heads
        head_dim = text.get("head_dim") or text["hidden_size"] // heads
    except (KeyError, TypeError, ZeroDivisionError):
        return None
    dtype = text.get("dtype") or text.get("torch_dtype") or model_config.get("torch_dtype")
    limits = [text[key] for key in config.VLLM_MAX_LENGTH_KEYS if text.get(key)]
    return ModelMemory(
        weights_bytes,
        2 * layers * kv_heads * head_dim,
        config.VLLM_DTYPE_BYTES.get(dtype, config.VLLM_DEFAULT_DTYPE_BYTES),
        min(limits, default=None),
    )


def colocated_vllm_share(phase: PhaseConfiguration) -> float:
    """The share of GPU memory the vLLM generating the Phase's rollouts takes, if one does."""
    settings = {
        **config.ALGORITHMS[phase.algorithm]["platform_settings"],
        **phase.settings.model_dump(),
    }
    return settings.get("vllm_gpu_memory_utilization", 0) if settings.get("use_vllm") else 0


def largest_calculated_share(gpu: Gpu, settings: Settings) -> float:
    """The most a default takes; only a value the user enters goes higher."""
    largest = settings.gpu_share_maximum
    if gpu.shared_with_system:
        largest = settings.gpu_share_maximum_shared_memory
    return min(largest, settings.gpu_budget) if settings.gpu_sharing else largest


def kv_cache_dtype_bytes(options: ServingOptions, model: ModelMemory) -> int:
    if options.kv_cache_dtype not in (None, "auto"):
        return 1
    return config.VLLM_DTYPE_BYTES.get(options.dtype, model.dtype_bytes)


def distill_serves_its_teacher(teacher: str, api_url: str | None) -> bool:
    return not (api_url or teacher.startswith("endpoint:"))


def evaluate_serves_its_model(model: str | None) -> bool:
    return not (model or "").startswith("endpoint:")


def pipeline_gpu_shares(request: dict) -> dict[str, float]:
    """Each enabled Stage of the resolved request -> the GPU Share its steps hold."""
    shares = {}
    distill, evaluate = request.get("distill"), request.get("evaluate")
    if distill and distill_serves_its_teacher(distill["teacher"], distill.get("api_url")):
        shares["distill"] = stored_serving_share(distill)
    for stage in config.TRAINING_STAGES:
        if request.get(stage):
            shares[stage] = request.get("gpu_share") or config.GPU_SHARE
    if evaluate and evaluate_serves_its_model(evaluate.get("model")):
        shares["evaluate"] = stored_serving_share(evaluate)
    return shares


def stored_serving_share(stage: dict) -> float:
    # Requests stored before #116 may leave it out, meaning the then default.
    serving = stage.get("serving") or {}
    return serving.get("gpu_memory_utilization") or config.SERVING_GPU_MEMORY_UTILIZATION
