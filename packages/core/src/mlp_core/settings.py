import math

from pydantic_settings import BaseSettings, SettingsConfigDict

from mlp_core import config


class Settings(BaseSettings):
    """Platform settings; defaults live in deploy/values.yaml, rendered into a ConfigMap."""

    model_config = SettingsConfigDict(extra="forbid")

    # The cluster's physical GPUs, counted by install.sh; every step and Endpoint uses one.
    gpu_count: int
    # Off if a machine has several GPUs, as install.sh decides: jobs then take whole GPUs.
    gpu_sharing: bool
    gpu_slice: float
    # What a calculated GPU Share counts in; see deploy/values.yaml.
    gpu_share_weights_factor: float
    gpu_share_vllm_reserve_gib: float
    gpu_share_shared_memory_reserve_gib: float
    gpu_share_max_model_len: int
    gpu_share_training_factor: float
    gpu_share_full_training_factor: float
    gpu_share_training_reserve_gib: float
    gpu_max_utilization: float
    sandbox_timeout_seconds: int
    sandbox_memory_mb: int
    sandbox_replicas: int
    sandbox_cpus_per_replica: int
    model_cache_size: str
    model_cache_high_water_mark: float
    object_store_size: str
    checkpoint_minutes: int
    speculate_dataloader_workers: int

    def gpu_request(self, share: float) -> int:
        """The `nvidia.com/gpu` a job asks for: the share's slices of one GPU, or the whole GPU."""
        return self.gpu_slices(share) if self.gpu_sharing else 1

    def gpu_slices(self, share: float) -> int:
        """The whole slices the share takes on one GPU."""
        # Rounded first, as 0.15 / 0.05 is a hair above 3 and must not become 4 slices.
        return math.ceil(round(share / self.gpu_slice, 6))

    def share_in_slices(self, share: float) -> float:
        """The share rounded up to whole slices."""
        return round(self.gpu_slices(share) * self.gpu_slice, 6)

    def gpus_of_request(self, request: int) -> float:
        """The GPUs a `nvidia.com/gpu` request adds up to: slices of one, or whole ones."""
        return request * self.gpu_slice if self.gpu_sharing else request

    def gpu_share_problem(self, share: float) -> str | None:
        """Why the GPU Share can never fit, if it can't."""
        return f"a GPU Share of {share} can never fit: it is above 1" if share > 1 else None


def size_in_bytes(quantity: str) -> int:
    """A Kubernetes size such as `100Gi`, in bytes."""
    number = quantity.rstrip("KMGTPi")
    return int(float(number) * config.QUANTITY_SUFFIXES[quantity[len(number) :]])
