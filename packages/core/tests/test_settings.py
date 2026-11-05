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


def test_settings_reject_a_phase_on_more_nodes_than_the_cluster_has(platform_settings):
    with pytest.raises(ValidationError, match="nodes_per_stage"):
        Settings(**{**platform_settings, "nodes_per_stage": 2, "node_count": 1})


def test_settings_reject_a_phase_needing_more_gpus_than_its_nodes_have(platform_settings):
    phase = {"node_count": 2, "nodes_per_stage": 2, "gpus_per_stage": 2, "gpu_count": 3}

    with pytest.raises(ValidationError, match=r"nodes_per_stage × gpus_per_stage \(4\)"):
        Settings(**{**platform_settings, **phase})


def test_a_phase_may_use_every_gpu_of_every_node(platform_settings):
    phase = {"node_count": 2, "nodes_per_stage": 2, "gpus_per_stage": 2, "gpu_count": 4}

    assert Settings(**{**platform_settings, **phase}).nodes_per_stage == 2


@pytest.mark.parametrize(
    ("share", "gpus", "slices"),
    [(0.3, 1, 6), (0.33, 1, 7), (0.15, 1, 3), (0.05, 1, 1), (1.0, 1, 20), (0.3, 2, 12)],
)
def test_a_gpu_share_is_rounded_up_to_slices_on_each_gpu(platform_settings, share, gpus, slices):
    assert Settings(**platform_settings).gpu_request(share, gpus) == slices


def test_jobs_request_whole_gpus_while_sharing_is_off(platform_settings):
    settings = Settings(**{**platform_settings, "gpu_sharing": False, "gpu_count": 2})

    assert settings.gpu_request(0.3, 2) == 2


def test_only_a_gpu_share_above_1_can_never_fit(platform_settings):
    settings = Settings(**platform_settings)

    assert settings.gpu_share_problem(settings.gpu_max_utilization + 0.1) is None
    assert settings.gpu_share_problem(1) is None
    assert "1.1" in settings.gpu_share_problem(1.1)
