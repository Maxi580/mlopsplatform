from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from mlp_core import config


class Settings(BaseSettings):
    """Platform settings; defaults live in deploy/values.yaml, rendered into a ConfigMap."""

    model_config = SettingsConfigDict(extra="forbid")

    gpu_count: int
    gpus_per_stage: int
    gpus_per_endpoint: int
    sandbox_timeout_seconds: int
    sandbox_memory_mb: int
    model_cache_size: str
    model_cache_high_water_mark: float
    object_store_size: str
    checkpoint_minutes: int
    speculate_dataloader_workers: int

    @model_validator(mode="after")
    def check_a_stage_fits_on_the_platform(self) -> "Settings":
        if self.gpus_per_stage > self.gpu_count:
            raise ValueError(
                f"gpus_per_stage ({self.gpus_per_stage}) exceeds gpu_count ({self.gpu_count})"
            )
        return self


def size_in_bytes(quantity: str) -> int:
    """A Kubernetes size such as `100Gi`, in bytes."""
    number = quantity.rstrip("KMGTPi")
    return int(float(number) * config.QUANTITY_SUFFIXES[quantity[len(number) :]])
