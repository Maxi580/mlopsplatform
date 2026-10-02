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


MODEL_FILES = {
    "config.json": b'{"model_type": "qwen2"}',
    "model.safetensors": b"0123456789",
    "original/params.json": b"{}",
}


def model_directory(tmp_path):
    directory = tmp_path / "my-model"
    for path, content in {**MODEL_FILES, ".git/HEAD": b"ref"}.items():
        (directory / path).parent.mkdir(parents=True, exist_ok=True)
        (directory / path).write_bytes(content)
    return directory


def answer_upload_start(fake_api, part_size):
    """Part URLs on the fake API, one per part of each file."""
    files = [
        {
            "path": path,
            "part_urls": [
                f"{fake_api.url}/parts/{path}/{number}"
                for number in range(1, max(1, -(-len(content) // part_size)) + 1)
            ],
        }
        for path, content in MODEL_FILES.items()
    ]
    started = {"id": "abc", "part_size_bytes": part_size, "files": files}
    fake_api.answers[api_paths.MODEL_UPLOADS] = (201, started)


def test_upload_sends_every_file_in_parts_then_completes(logged_in, fake_api, tmp_path):
    answer_upload_start(fake_api, part_size=4)
    complete = api_paths.MODEL_UPLOAD_COMPLETE.format(id="abc")
    fake_api.answers[complete] = (201, {"name": "my-model", "version": 3})

    result = mlp_models(
        "upload", str(model_directory(tmp_path)), "--name", "my-model", "--base", "hf:org/base"
    )

    assert result.exit_code == 0, result.output
    assert "my-model@3" in result.output
    [(_, started), (completed, _)] = fake_api.received
    assert started == {
        "name": "my-model",
        "files": [{"path": path, "size_bytes": len(c)} for path, c in MODEL_FILES.items()],
        "base": "hf:org/base",
        "tool_parser": None,
    }
    assert completed == complete
    sent = {}
    for url, authorization, body in fake_api.parts:
        assert authorization is None
        path = url.removeprefix("/parts/").rsplit("/", 1)[0]
        sent[path] = sent.get(path, b"") + body
    assert sent == MODEL_FILES


def test_a_refused_upload_shows_the_reason(logged_in, fake_api, tmp_path):
    detail = "Rejected my-model: config.json is missing"
    fake_api.answers[api_paths.MODEL_UPLOADS] = (422, {"detail": detail})

    result = mlp_models("upload", str(model_directory(tmp_path)), "--name", "my-model")

    assert result.exit_code == 1
    assert detail in result.output
    assert fake_api.parts == []


def test_download_saves_every_file_of_the_version(logged_in, fake_api, tmp_path):
    files = [
        {"path": path, "size_bytes": len(content), "url": f"{fake_api.url}/objects/{path}"}
        for path, content in MODEL_FILES.items()
    ]
    path = api_paths.MODEL_VERSION_FILES.format(name="qwen-sft", version=2)
    fake_api.answers[path] = (200, {"files": files})
    for file, content in MODEL_FILES.items():
        fake_api.answers[f"/objects/{file}"] = (200, content)

    result = mlp_models("download", "qwen-sft@2", "-o", str(tmp_path / "out"))

    assert result.exit_code == 0, result.output
    for file, content in MODEL_FILES.items():
        assert (tmp_path / "out" / file).read_bytes() == content
