import logging
import sys
from types import SimpleNamespace

import pytest

from mlp_stages.main import main
from mlp_stages.operations import fetch

HF_TOKEN = "hf_" + "s3cr3tT0ken" * 4
REPO = "Qwen/Qwen2.5-0.5B-Instruct"
COMMIT = "7ae557604adf67be50417f59c2c2f167def9a775"
BASE_MODEL = f"hf:{REPO}@{COMMIT}"
# Repo -> the sizes of its files on Hugging Face.
HUB_FILES = {REPO: [400, 100]}


class FakeHfApi:
    def __init__(self, token):
        pass

    def model_info(self, repo, revision, files_metadata):
        return SimpleNamespace(siblings=[SimpleNamespace(size=size) for size in HUB_FILES[repo]])


@pytest.fixture
def model_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch, "HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(fetch, "HfApi", FakeHfApi)
    return tmp_path / "hub"


@pytest.fixture
def downloads(monkeypatch, model_cache):
    downloads = []
    monkeypatch.setattr(
        fetch, "snapshot_download", lambda repo, **options: downloads.append((repo, options))
    )
    # main() wraps stdout and stderr; monkeypatch puts the originals back.
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    return downloads


def cache_base_model(model_cache, repo, commit, size_bytes):
    snapshot = model_cache / f"models--{repo.replace('/', '--')}" / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"x" * size_bytes)


def mlp_stage(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["mlp-stage", *args])
    main()


def test_fetch_downloads_the_base_model_at_its_pinned_commit(monkeypatch, downloads):
    monkeypatch.setenv("HF_TOKEN", HF_TOKEN)

    mlp_stage(monkeypatch, "fetch", BASE_MODEL, "100Gi")

    assert downloads == [
        (
            "Qwen/Qwen2.5-0.5B-Instruct",
            {"revision": "7ae557604adf67be50417f59c2c2f167def9a775", "token": HF_TOKEN},
        )
    ]


def test_fetch_downloads_anonymously_without_a_token(monkeypatch, downloads):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    mlp_stage(monkeypatch, "fetch", BASE_MODEL, "100Gi")

    assert downloads[0][1]["token"] is None


def test_secret_values_are_redacted_from_the_step_output(monkeypatch, downloads, capsys):
    monkeypatch.setenv("HF_TOKEN", HF_TOKEN)

    def leak(repo, **options):
        print(f"token={options['token']}")
        logging.getLogger("huggingface_hub").warning("header Bearer %s", options["token"])
        raise RuntimeError(f"401 for token {options['token']}")

    monkeypatch.setattr(fetch, "snapshot_download", leak)

    with pytest.raises(RuntimeError):
        mlp_stage(monkeypatch, "fetch", BASE_MODEL, "100Gi")

    output = capsys.readouterr()
    assert HF_TOKEN not in output.out + output.err
    assert "token=***" in output.out
    assert "Bearer ***" in output.err


def test_cleanup_runs(monkeypatch, downloads):
    mlp_stage(monkeypatch, "cleanup")


def test_fetch_fails_clearly_when_the_download_does_not_fit(monkeypatch, downloads, model_cache):
    cache_base_model(model_cache, "org/other", "1" * 40, 600)

    with pytest.raises(RuntimeError, match="needs 500 bytes but the Model Cache has 400 of 1000"):
        mlp_stage(monkeypatch, "fetch", BASE_MODEL, "1000")

    assert downloads == []


def test_files_already_cached_need_no_room(monkeypatch, downloads, model_cache):
    cache_base_model(model_cache, "org/other", "1" * 40, 400)
    cache_base_model(model_cache, REPO, COMMIT, 400)

    mlp_stage(monkeypatch, "fetch", BASE_MODEL, "1000")

    assert len(downloads) == 1
