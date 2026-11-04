import os

from mlp_core import config
from mlp_core.pipeline_request.schema import PhaseConfiguration


def cap_gpu_memory(phase: PhaseConfiguration | None = None) -> None:
    """Keeps this process's PyTorch to the step's GPU Share; call it before any model loads."""
    # 1. No GPU Share while sharing is off: the step has whole GPUs.
    if config.GPU_SHARE_ENV not in os.environ:
        return
    share = float(os.environ[config.GPU_SHARE_ENV])

    # 2. Colocated vLLM takes its memory out of the same share.
    vllm = colocated_vllm_share(phase) if phase else 0
    if share <= vllm:
        raise SystemExit(
            f"The GPU Share of {share} leaves no memory for training beside colocated vLLM's "
            f"vllm_gpu_memory_utilization of {vllm}; raise `gpu_share` or lower it."
        )

    # 3. The limit is per process, and vLLM allocates inside this one, so it holds both.
    import torch

    torch.cuda.set_per_process_memory_fraction(share)


def colocated_vllm_share(phase: PhaseConfiguration) -> float:
    """The share of GPU memory the vLLM generating the Phase's rollouts takes, if one does."""
    settings = {
        **config.ALGORITHMS[phase.algorithm]["platform_settings"],
        **phase.settings.model_dump(),
    }
    return settings.get("vllm_gpu_memory_utilization", 0) if settings.get("use_vllm") else 0
