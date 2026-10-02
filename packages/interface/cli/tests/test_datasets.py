import pytest
from typer.testing import CliRunner

from mlp_cli.main import app
from mlp_core import api_paths

from .conftest import TOKEN, write_profile

CONTENT = b'{"text": "hello"}\n'
UPLOAD_PATH = api_paths.DATASET_VERSIONS.format(name="chat")
DOWNLOAD_PATH = api_paths.DATASET_DOWNLOAD.format(name="chat", version=2)


@pytest.fixture
def logged_in(home, fake_api, platform_ca):
    write_profile(home, fake_api.url, platform_ca)
    (home / ".mlp" / "token").write_text(TOKEN)


def mlp_datasets(*args):
    return CliRunner().invoke(app, ["datasets", *args])


def test_upload_sends_the_file_and_prints_the_new_version(logged_in, home, fake_api):
    (home / "chat.jsonl").write_bytes(CONTENT)
    fake_api.answers[UPLOAD_PATH] = (201, {"version": 2, "size_bytes": 18, "row_format": "text"})

    result = mlp_datasets("upload", str(home / "chat.jsonl"), "--name", "chat")

    assert result.exit_code == 0, result.output
    assert fake_api.received == [(UPLOAD_PATH, CONTENT)]
    assert "chat@2" in result.output


def test_a_rejected_upload_prints_the_reason(logged_in, home, fake_api):
    (home / "chat.jsonl").write_bytes(b"{not json\n")
    fake_api.answers[UPLOAD_PATH] = (422, {"detail": "Rejected chat: line 1: is not JSON"})

    result = mlp_datasets("upload", str(home / "chat.jsonl"), "--name", "chat")

    assert result.exit_code == 1
    assert "line 1: is not JSON" in result.output


def test_datasets_lists_every_version_with_its_size(logged_in, fake_api):
    versions = [
        {"version": 1, "size_bytes": 1234, "row_format": "messages"},
        {"version": 3, "size_bytes": 5, "row_format": "messages"},
    ]
    fake_api.answers[api_paths.DATASETS] = (200, [{"name": "chat", "versions": versions}])

    result = mlp_datasets()

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert "chat@1" in lines[1] and "1,234 bytes" in lines[1] and "messages" in lines[1]
    assert "chat@3" in lines[2]


def test_download_saves_the_version_from_its_presigned_url(logged_in, home, fake_api):
    fake_api.answers[DOWNLOAD_PATH] = (200, {"url": f"{fake_api.url}/object?signature=x"})
    fake_api.answers["/object?signature=x"] = (200, CONTENT)

    result = mlp_datasets("download", "chat@2", "-o", str(home / "out.jsonl"))

    assert result.exit_code == 0, result.output
    assert (home / "out.jsonl").read_bytes() == CONTENT
    # The signature is the object store's only credential; the login token stays with the API.
    assert fake_api.received[-1] == ("/object?signature=x", None)


def test_download_names_the_file_after_the_version_by_default(
    logged_in, home, fake_api, monkeypatch
):
    fake_api.answers[DOWNLOAD_PATH] = (200, {"url": f"{fake_api.url}/object"})
    fake_api.answers["/object"] = (200, CONTENT)
    monkeypatch.chdir(home)

    mlp_datasets("download", "chat@2")

    assert (home / "chat-2.jsonl").read_bytes() == CONTENT


def test_delete_removes_the_named_version(logged_in, fake_api):
    path = api_paths.DATASET_VERSION.format(name="chat", version=2)
    fake_api.answers[path] = (204, b"")

    result = mlp_datasets("delete", "chat@2")

    assert result.exit_code == 0, result.output
    assert fake_api.received == [(path, f"Bearer {TOKEN}")]


@pytest.mark.parametrize("command", ["download", "delete"])
def test_a_version_is_required(logged_in, fake_api, command):
    result = mlp_datasets(command, "chat")

    assert result.exit_code != 0
    assert "name@version" in result.output
    assert fake_api.received == []
