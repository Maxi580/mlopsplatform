import logging
import sys

import pytest

from mlp_stages import fetch
from mlp_stages.main import main

HF_TOKEN = "hf_" + "s3cr3tT0ken" * 4
BASE_MODEL = "hf:Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775"


@pytest.fixture
def downloads(monkeypatch):
    downloads = []
    monkeypatch.setattr(
        fetch, "snapshot_download", lambda repo, **options: downloads.append((repo, options))
    )
    # main() wraps stdout and stderr; monkeypatch puts the originals back.
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    return downloads


def mlp_stage(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["mlp-stage", *args])
    main()


def test_fetch_downloads_the_base_model_at_its_pinned_commit(monkeypatch, downloads):
    monkeypatch.setenv("HF_TOKEN", HF_TOKEN)

    mlp_stage(monkeypatch, "fetch", BASE_MODEL)

    assert downloads == [
        (
            "Qwen/Qwen2.5-0.5B-Instruct",
            {"revision": "7ae557604adf67be50417f59c2c2f167def9a775", "token": HF_TOKEN},
        )
    ]


def test_fetch_downloads_anonymously_without_a_token(monkeypatch, downloads):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    mlp_stage(monkeypatch, "fetch", BASE_MODEL)

    assert downloads[0][1]["token"] is None


def test_secret_values_are_redacted_from_the_step_output(monkeypatch, downloads, capsys):
    monkeypatch.setenv("HF_TOKEN", HF_TOKEN)

    def leak(repo, **options):
        print(f"token={options['token']}")
        logging.getLogger("huggingface_hub").warning("header Bearer %s", options["token"])
        raise RuntimeError(f"401 for token {options['token']}")

    monkeypatch.setattr(fetch, "snapshot_download", leak)

    with pytest.raises(RuntimeError):
        mlp_stage(monkeypatch, "fetch", BASE_MODEL)

    output = capsys.readouterr()
    assert HF_TOKEN not in output.out + output.err
    assert "token=***" in output.out
    assert "Bearer ***" in output.err


def test_cleanup_runs(monkeypatch, downloads):
    mlp_stage(monkeypatch, "cleanup")
