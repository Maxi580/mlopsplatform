from pathlib import Path

import pytest
import yaml


@pytest.fixture
def platform_settings() -> dict:
    return yaml.safe_load((Path(__file__).parent / "deploy" / "values.yaml").read_text())[
        "settings"
    ]


@pytest.fixture
def settings_configmap_env(monkeypatch, platform_settings):
    for key, value in platform_settings.items():
        monkeypatch.setenv(key, str(value))
