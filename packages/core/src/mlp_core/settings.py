from pydantic_settings import BaseSettings, SettingsConfigDict


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
