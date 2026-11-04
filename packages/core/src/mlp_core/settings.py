import math

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from mlp_core import config


class Settings(BaseSettings):
    """Platform settings; defaults live in deploy/values.yaml, rendered into a ConfigMap."""

    model_config = SettingsConfigDict(extra="forbid")

    # The cluster's GPU machines and physical GPUs, counted by install.sh; machines are identical.
    node_count: int
    gpu_count: int
    # Off if a machine has several GPUs, as install.sh decides: jobs then take whole GPUs.
    gpu_sharing: bool
    gpu_budget: float
    gpu_slice: float
    # GPUs per node for a Stage; a finetune Phase trains on nodes_per_stage nodes, the rest on one.
    gpus_per_stage: int
    nodes_per_stage: int
    gpus_per_endpoint: int
    sandbox_timeout_seconds: int
    sandbox_memory_mb: int
    sandbox_replicas: int
    sandbox_cpus_per_replica: int
    model_cache_size: str
    model_cache_high_water_mark: float
    object_store_size: str
    checkpoint_minutes: int
    speculate_dataloader_workers: int

    @model_validator(mode="after")
    def check_a_stage_and_an_endpoint_fit_on_the_platform(self) -> "Settings":
        for name in ("gpus_per_stage", "gpus_per_endpoint"):
            if getattr(self, name) > self.gpu_count:
                raise ValueError(
                    f"{name} ({getattr(self, name)}) exceeds gpu_count ({self.gpu_count})"
                )
        if self.nodes_per_stage > self.node_count:
            raise ValueError(
                f"nodes_per_stage ({self.nodes_per_stage}) exceeds node_count ({self.node_count})"
            )
        if self.phase_gpus > self.gpu_count:
            raise ValueError(
                f"nodes_per_stage × gpus_per_stage ({self.phase_gpus}) exceeds gpu_count "
                f"({self.gpu_count})"
            )
        return self

    @property
    def phase_gpus(self) -> int:
        """The GPUs a finetune Phase trains on, across its nodes."""
        return self.nodes_per_stage * self.gpus_per_stage

    def gpu_request(self, share: float, gpus: int) -> int:
        """The `nvidia.com/gpu` a job asks for on `gpus` GPUs: the share's slices, or whole GPUs."""
        if not self.gpu_sharing:
            return gpus
        # Rounded first, as 0.15 / 0.05 is a hair above 3 and must not become 4 slices.
        return math.ceil(round(share / self.gpu_slice, 6)) * gpus

    def gpus_of_request(self, request: int) -> float:
        """The GPUs a `nvidia.com/gpu` request adds up to: slices of one, or whole ones."""
        return request * self.gpu_slice if self.gpu_sharing else request

    def gpu_share_problem(self, share: float) -> str | None:
        """Why the GPU Share can never fit, if it can't."""
        if self.gpu_sharing and share > self.gpu_budget:
            return f"a GPU Share of {share} can never fit: the budget is {self.gpu_budget} of a GPU"
        return None


def size_in_bytes(quantity: str) -> int:
    """A Kubernetes size such as `100Gi`, in bytes."""
    number = quantity.rstrip("KMGTPi")
    return int(float(number) * config.QUANTITY_SUFFIXES[quantity[len(number) :]])
