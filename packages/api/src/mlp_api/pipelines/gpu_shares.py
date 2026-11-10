from functools import cache

from mlp_api.models.mlflow import MLflow
from mlp_api.models.model_memory import find_model_memory
from mlp_api.pipelines.hugging_face import HuggingFace
from mlp_api.storage.object_store import ObjectStore
from mlp_core import config
from mlp_core.endpoint_spec import ServingOptions
from mlp_core.gpu_share import (
    Gpu,
    colocated_vllm_share,
    distill_serves_its_teacher,
    evaluate_serves_its_model,
    training_gpu_share,
    with_gpu_share,
)
from mlp_core.pipeline_request.schema import PipelineRequest
from mlp_core.settings import Settings


def resolve_gpu_shares(
    request: PipelineRequest,
    gpu: Gpu | None,
    token: str | None,
    hugging_face: HuggingFace,
    model_registry: MLflow,
    object_store: ObjectStore,
    settings: Settings,
) -> None:
    """Pins every GPU Share and `max_model_len` into the request, keeping what the user entered."""
    # Several steps often read the same model.
    memory_of = cache(
        lambda model: find_model_memory(model, token, hugging_face, model_registry, object_store)
    )

    # 1. Each step's own vLLM.
    distill, evaluate = request.distill, request.evaluate
    if distill and distill_serves_its_teacher(distill.teacher, distill.api_url):
        distill.serving = with_gpu_share(
            distill.serving or ServingOptions(), memory_of(distill.teacher), gpu, settings
        )
    if evaluate and evaluate_serves_its_model(evaluate.model):
        memory = memory_of(model_behind_output(request, evaluate.model))
        evaluate.serving = with_gpu_share(
            evaluate.serving or ServingOptions(), memory, gpu, settings
        )

    # 2. The Pipeline's, for its largest step.
    stages = (getattr(request, stage) for stage in config.TRAINING_STAGES)
    if request.gpu_share is None and any(stages):
        request.gpu_share = largest_step_share(request, gpu, memory_of, settings)


def model_behind_output(request: PipelineRequest, model: str) -> str:
    """The model a Stage's output is sized as, since it doesn't exist yet; else the model itself."""
    if model == config.QUANTIZE_OUTPUT:
        model = request.quantize.model
    if model == config.FINETUNE_OUTPUT:
        model = request.finetune.starting_model
    return model


def largest_step_share(
    request: PipelineRequest, gpu: Gpu | None, memory_of, settings: Settings
) -> float:
    """What the Pipeline's largest training step needs; the fixed share if a size is unknown."""
    # 1. What each step trains or loads, with its method, Teacher and colocated vLLM.
    sweep, finetune = request.sweep, request.finetune
    steps = []
    if sweep:
        steps.append((sweep.model, sweep.method, sweep.teacher, colocated_vllm_share(sweep)))
    for phase in finetune.phases if finetune else []:
        colocated = colocated_vllm_share(phase)
        steps.append((finetune.starting_model, phase.method, phase.teacher, colocated))
    for stage in (request.quantize, request.speculate):
        if stage:
            steps.append((model_behind_output(request, stage.model), None, None, 0))

    # 2. The largest of them; one unknown size can't be left out, so it means the fixed share.
    shares = [step_share(*step, gpu, memory_of, settings) for step in steps]
    return config.GPU_SHARE if None in shares else max(shares)


def step_share(
    model: str,
    method: str | None,
    teacher: str | None,
    colocated: float,
    gpu: Gpu | None,
    memory_of,
    settings: Settings,
) -> float | None:
    """The GPU Share of one training step, or None if a size is unknown."""
    weights_bytes = memory_of(model).weights_bytes
    teacher_bytes = memory_of(teacher).weights_bytes if teacher else 0
    if gpu is None or weights_bytes is None or teacher_bytes is None:
        return None
    return training_gpu_share(weights_bytes, method, teacher_bytes, colocated, gpu, settings)
