from dataclasses import dataclass

from mlp_core import config
from mlp_core.endpoint_spec import ServingOptions
from mlp_core.pipeline_request.schema import PhaseConfiguration
from mlp_core.settings import Settings


@dataclass(frozen=True)
class Gpu:
    memory_bytes: int
    shared_with_system: bool


@dataclass(frozen=True)
class ModelMemory:
    """What a model's files say, each None if they don't."""

    weights_bytes: int | None
    # The longest prompt plus reply the model allows.
    max_length: int | None


def with_gpu_share(
    options: ServingOptions,
    model: ModelMemory,
    gpu: Gpu | None,
    gpus: int,
    settings: Settings,
) -> ServingOptions:
    """The options with a GPU Share and length: the user's, else calculated, else the fixed ones."""
    # 1. A very long model limit would often fail at startup, so it is cut; unknown, vLLM decides.
    length = options.max_model_len
    if length is None and model.max_length is not None:
        length = min(model.max_length, settings.gpu_share_max_model_len)

    # 2. A value the user entered wins.
    share = options.gpu_memory_utilization
    if share is None:
        share = vllm_gpu_share(model.weights_bytes, gpu, gpus, settings)
    return options.model_copy(update={"gpu_memory_utilization": share, "max_model_len": length})


def vllm_gpu_share(
    weights_bytes: int | None, gpu: Gpu | None, gpus: int, settings: Settings
) -> float:
    """The GPU Share to serve the weights, across `gpus`; the fixed one if a size is unknown."""
    if weights_bytes is None or gpu is None:
        return config.SERVING_GPU_MEMORY_UTILIZATION
    reserve = settings.gpu_share_vllm_reserve_gib
    if gpu.shared_with_system:
        reserve += settings.gpu_share_shared_memory_reserve_gib
    weights = settings.gpu_share_weights_factor * weights_bytes / gpus
    need = weights + reserve * config.QUANTITY_SUFFIXES["Gi"]
    return min(settings.share_in_slices(need / gpu.memory_bytes), settings.gpu_max_utilization)


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
        settings.gpu_max_utilization,
    )


def model_length_limit(model_config: dict) -> int | None:
    """The longest prompt plus reply the config.json allows, if it says."""
    # Multimodal models nest their language model's config.
    text = model_config.get("text_config") or model_config
    limits = [text[key] for key in config.VLLM_MAX_LENGTH_KEYS if text.get(key)]
    return min(limits, default=None)


def colocated_vllm_share(phase: PhaseConfiguration) -> float:
    """The share of GPU memory the vLLM generating the Phase's rollouts takes, if one does."""
    settings = {
        **config.ALGORITHMS[phase.algorithm]["platform_settings"],
        **phase.settings.model_dump(),
    }
    return settings.get("vllm_gpu_memory_utilization", 0) if settings.get("use_vllm") else 0


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
