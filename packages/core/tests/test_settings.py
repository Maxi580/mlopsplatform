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


@pytest.mark.parametrize(
    ("share", "slices"), [(0.3, 6), (0.33, 7), (0.15, 3), (0.05, 1), (1.0, 20)]
)
def test_a_gpu_share_is_rounded_up_to_slices(platform_settings, share, slices):
    assert Settings(**platform_settings).gpu_request(share) == slices


def test_jobs_request_a_whole_gpu_while_sharing_is_off(platform_settings):
    settings = Settings(**{**platform_settings, "gpu_sharing": False, "gpu_count": 2})

    assert settings.gpu_request(0.3) == 1


def test_only_a_gpu_share_above_1_can_never_fit(platform_settings):
    settings = Settings(**platform_settings)

    assert settings.gpu_share_problem(settings.gpu_max_utilization + 0.1) is None
    assert settings.gpu_share_problem(1) is None
    assert "1.1" in settings.gpu_share_problem(1.1)
