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
