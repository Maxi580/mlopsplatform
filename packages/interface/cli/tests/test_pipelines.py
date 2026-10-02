import pytest
from huggingface_hub import constants as hugging_face_constants
from typer.testing import CliRunner

from mlp_cli.main import app
from mlp_core import api_paths

from .conftest import TOKEN, write_profile
from .test_validate import HF_TOKEN, PROFILE

PROFILE_WITHOUT_SECRETS = {key: value for key, value in PROFILE.items() if key != "secrets"}
OTHER_TOKEN = "hf_" + "y" * 34
DOWNLOAD_BYTES, CACHED_BYTES = int(9.4 * 2**30), int(2.1 * 2**30)


@pytest.fixture
def hf_login(home, monkeypatch):
    """Where `hf auth login` keeps its token; empty unless a test logs in."""
    path = home / "hf-token"
    monkeypatch.setattr(hugging_face_constants, "HF_TOKEN_PATH", str(path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return path


@pytest.fixture
def logged_in(home, fake_api, platform_ca, hf_login):
    (home / ".mlp" / "token").write_text(TOKEN)
    fake_api.answers[api_paths.PIPELINES] = (202, {"id": 7, "request": {}})


def mlp(*args, input=None):
    return CliRunner().invoke(app, list(args), input=input)


def sent_secrets(fake_api):
    [(path, submission)] = fake_api.received
    assert path == api_paths.PIPELINES
    return submission["secrets"]


def test_run_submits_and_prints_the_pipeline_id(logged_in, home, fake_api, platform_ca):
    write_profile(home, fake_api.url, platform_ca, **PROFILE)

    result = mlp("run", "--finetune", "sft")

    assert result.exit_code == 0, result.output
    assert "Pipeline 7" in result.output
    assert sent_secrets(fake_api) == {"hf_token": HF_TOKEN}


def test_run_takes_the_hf_token_from_the_environment(
    logged_in, home, fake_api, platform_ca, monkeypatch
):
    write_profile(home, fake_api.url, platform_ca, **PROFILE_WITHOUT_SECRETS)
    monkeypatch.setenv("HF_TOKEN", OTHER_TOKEN)

    mlp("run", "--finetune", "sft")

    assert sent_secrets(fake_api) == {"hf_token": OTHER_TOKEN}


def test_run_takes_the_hf_auth_login_token(logged_in, home, fake_api, platform_ca, hf_login):
    write_profile(home, fake_api.url, platform_ca, **PROFILE_WITHOUT_SECRETS)
    hf_login.write_text(OTHER_TOKEN)

    mlp("run", "--finetune", "sft")

    assert sent_secrets(fake_api) == {"hf_token": OTHER_TOKEN}


def test_run_asks_for_the_hf_token_without_echoing_it(logged_in, home, fake_api, platform_ca):
    write_profile(home, fake_api.url, platform_ca, **PROFILE_WITHOUT_SECRETS)

    result = mlp("run", "--finetune", "sft", input=f"{OTHER_TOKEN}\n")

    assert sent_secrets(fake_api) == {"hf_token": OTHER_TOKEN}
    assert OTHER_TOKEN not in result.output


def test_run_without_any_hf_token_sends_none(logged_in, home, fake_api, platform_ca):
    write_profile(home, fake_api.url, platform_ca, **PROFILE_WITHOUT_SECRETS)

    mlp("run", "--finetune", "sft", input="\n")

    assert sent_secrets(fake_api) == {}


def test_run_prints_each_rejection_with_its_path(logged_in, home, fake_api, platform_ca):
    write_profile(home, fake_api.url, platform_ca, **PROFILE)
    detail = [{"loc": ["finetune", "base_model"], "msg": "is missing"}]
    fake_api.answers[api_paths.PIPELINES] = (422, {"detail": detail})

    result = mlp("run", "--finetune", "sft")

    assert result.exit_code == 1
    assert "finetune.base_model: is missing" in result.output


def test_ls_shows_owner_status_stages_and_links(logged_in, home, fake_api, platform_ca):
    write_profile(home, fake_api.url, platform_ca, **PROFILE)
    pipeline = {
        "id": 7,
        "name": "qwen-sft",
        "owner": "shared",
        "status": "waiting for GPU",
        "stages": ["finetune"],
        "created_at": "2026-10-02T12:00:00+00:00",
        "kubeflow_run_url": "/pipeline/#/runs/details/run-1",
        "mlflow_run_url": "https://mlflow.test/#/runs/abc",
    }
    fake_api.answers[api_paths.PIPELINES] = (200, [pipeline])

    result = mlp("ls")

    assert result.exit_code == 0, result.output
    row = result.output.splitlines()[1]
    for value in ("7", "qwen-sft", "shared", "waiting for GPU", "finetune"):
        assert value in row
    assert f"{fake_api.url}/pipeline/#/runs/details/run-1" in result.output
    assert "https://mlflow.test/#/runs/abc" in result.output


def test_cancel_cancels_the_pipeline(logged_in, home, fake_api, platform_ca):
    write_profile(home, fake_api.url, platform_ca, **PROFILE)
    path = api_paths.CANCEL_PIPELINE.format(id=7)
    fake_api.answers[path] = (200, {"id": 7, "status": "cancelled"})

    result = mlp("cancel", "7")

    assert result.exit_code == 0, result.output
    assert fake_api.received[0][0] == path
    assert "cancelled" in result.output


def test_rerun_submits_the_stored_request_with_fresh_secrets(
    logged_in, home, fake_api, platform_ca
):
    write_profile(home, fake_api.url, platform_ca, **PROFILE_WITHOUT_SECRETS)
    stored = {"name": "qwen-sft", "finetune": {"base_model": "hf:Qwen/Qwen3-0.6B@abc"}}
    fake_api.answers[api_paths.PIPELINE.format(id=3)] = (200, {"id": 3, "request": stored})

    result = mlp("rerun", "3", input=f"{OTHER_TOKEN}\n")

    assert result.exit_code == 0, result.output
    [(path, submission)] = [r for r in fake_api.received if r[0] == api_paths.PIPELINES]
    assert submission == {"request": stored, "secrets": {"hf_token": OTHER_TOKEN}}
    assert "Pipeline 7" in result.output


def test_run_dry_run_prints_the_downloads_without_submitting(
    logged_in, home, fake_api, platform_ca
):
    write_profile(home, fake_api.url, platform_ca, **PROFILE)
    downloads = [
        {"kind": "base_model", "ref": "hf:Qwen/a@1", "bytes": DOWNLOAD_BYTES, "cached": False},
        {"kind": "base_model", "ref": "hf:Qwen/b@2", "bytes": CACHED_BYTES, "cached": True},
    ]
    totals = {"download_bytes": DOWNLOAD_BYTES, "cached_bytes": CACHED_BYTES}
    answer = {"request": {}, "downloads": downloads, **totals}
    fake_api.answers[api_paths.VALIDATE_PIPELINE] = (200, answer)

    result = mlp("run", "--finetune", "sft", "--dry-run")

    assert result.exit_code == 0, result.output
    [(path, _)] = fake_api.received
    assert path == api_paths.VALIDATE_PIPELINE
    assert "base_model  hf:Qwen/a@1  9.4 GB" in result.output
    assert "base_model  hf:Qwen/b@2  2.1 GB  cached" in result.output
    assert "9.4 GB to download, 2.1 GB already cached" in result.output
