import yaml
from typer.testing import CliRunner

from mlp_cli.main import app
from mlp_core import api_paths

from .conftest import write_profile

STANDARD = {"name": "qwen-sft", "finetune": {"phases": [{"settings": {"learning_rate": 1e-4}}]}}


def test_standard_prints_the_apis_standard_request_as_yaml(home, fake_api, platform_ca):
    write_profile(home, fake_api.url, platform_ca)
    fake_api.answers[api_paths.STANDARD_PIPELINE] = (200, STANDARD)

    result = CliRunner().invoke(app, ["standard"])

    assert result.exit_code == 0, result.output
    assert yaml.safe_load(result.output) == STANDARD
