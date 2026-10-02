from typer.testing import CliRunner

from mlp_cli.main import app
from mlp_core import api_paths

from .conftest import TOKEN

VERSION_PATH = api_paths.MODEL_VERSION.format(name="qwen-sft", version=2)


def mlp_models(*args):
    return CliRunner().invoke(app, ["models", *args])


def test_models_lists_every_version_with_its_size_and_lineage(logged_in, fake_api):
    tags = {"weights": "adapter", "base_model": "hf:Qwen/Qwen3@abc", "pipeline": "7"}
    versions = [{"version": 2, "size_bytes": 1234, "tags": tags}]
    fake_api.answers[api_paths.MODELS] = (200, [{"name": "qwen-sft", "versions": versions}])

    result = mlp_models()

    assert result.exit_code == 0, result.output
    row = result.output.splitlines()[1]
    for text in ("qwen-sft@2", "1,234 bytes", "adapter", "hf:Qwen/Qwen3@abc", "#7"):
        assert text in row


def test_delete_removes_the_named_version(logged_in, fake_api):
    fake_api.answers[VERSION_PATH] = (204, b"")

    result = mlp_models("delete", "qwen-sft@2")

    assert result.exit_code == 0, result.output
    assert fake_api.received == [(VERSION_PATH, f"Bearer {TOKEN}")]


def test_a_refused_delete_names_the_pipeline_using_it(logged_in, fake_api):
    detail = "qwen-sft@2 is used by Pipeline qwen-sft (#7)"
    fake_api.answers[VERSION_PATH] = (409, {"detail": detail})

    result = mlp_models("delete", "qwen-sft@2")

    assert result.exit_code == 1
    assert detail in result.output
