import pytest
from pydantic import ValidationError

from mlp_core.settings import Settings


def test_settings_load_from_the_configmap_environment(settings_configmap_env):
    settings = Settings()

    assert settings.checkpoint_minutes == 30
    assert settings.model_cache_high_water_mark == 0.9


def test_settings_reject_keys_unknown_to_the_settings_module(platform_settings):
    with pytest.raises(ValidationError):
        Settings(**platform_settings, unknown_setting=1)


def test_settings_reject_a_stage_needing_more_gpus_than_the_platform_has(platform_settings):
    with pytest.raises(ValidationError, match="gpus_per_stage"):
        Settings(**{**platform_settings, "gpus_per_stage": 2, "gpu_count": 1})


def test_settings_reject_an_endpoint_needing_more_gpus_than_the_platform_has(platform_settings):
    with pytest.raises(ValidationError, match="gpus_per_endpoint"):
        Settings(**{**platform_settings, "gpus_per_endpoint": 2, "gpu_count": 1})
