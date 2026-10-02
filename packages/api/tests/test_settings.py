from mlp_core import api_paths


def test_the_platform_settings_tell_users_how_many_gpus_there_are(logged_in_api, platform_settings):
    response = logged_in_api.get(api_paths.SETTINGS)

    assert response.status_code == 200
    assert response.json()["gpu_count"] == platform_settings["gpu_count"]


def test_the_platform_settings_require_login(api):
    assert api.get(api_paths.SETTINGS).status_code == 401
