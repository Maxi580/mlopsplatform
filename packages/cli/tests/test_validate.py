import pytest
from typer.testing import CliRunner

from mlp_cli.main import app
from mlp_core import api_paths

from .conftest import write_profile

HF_TOKEN = "hf_" + "x" * 34
PROFILE = {
    "name": "qwen-sft",
    "secrets": {"hf_token": HF_TOKEN},
    "finetune": {
        "base_model": "hf:Qwen/Qwen2.5-0.5B-Instruct",
        "phases": {
            "sft": {"dataset": "dataset:chat"},
            "sft-long": {
                "algorithm": "sft",
                "dataset": "dataset:chat",
                "settings": {"num_train_epochs": 9},
            },
        },
    },
    "evaluate": {"targets": ["@finetune"]},
}


@pytest.fixture
def profile(home, fake_api, platform_ca):
    write_profile(home, fake_api.url, platform_ca, **PROFILE)
    fake_api.answers[api_paths.VALIDATE_PIPELINE] = (200, {"request": {"name": "resolved"}})


def mlp_validate(*args):
    return CliRunner().invoke(app, ["validate", *args])


def sent_submission(fake_api):
    [(path, submission)] = fake_api.received
    assert path == api_paths.VALIDATE_PIPELINE
    return submission


def test_validate_sends_only_the_named_stages_and_phases(profile, fake_api):
    result = mlp_validate("--finetune", "sft-long")

    assert result.exit_code == 0, result.output
    request = sent_submission(fake_api)["request"]
    assert request == {
        "name": "qwen-sft",
        "finetune": {
            "base_model": "hf:Qwen/Qwen2.5-0.5B-Instruct",
            "phases": [
                {"algorithm": "sft", "dataset": "dataset:chat", "settings": {"num_train_epochs": 9}}
            ],
        },
    }


def test_a_phase_named_like_its_algorithm_needs_no_algorithm_field(profile, fake_api):
    mlp_validate("--finetune", "sft")

    phase = sent_submission(fake_api)["request"]["finetune"]["phases"][0]
    assert phase == {"algorithm": "sft", "dataset": "dataset:chat"}


def test_secrets_travel_beside_the_request_never_inside_it(profile, fake_api):
    mlp_validate("--finetune", "sft")

    submission = sent_submission(fake_api)
    assert submission["secrets"] == {"hf_token": HF_TOKEN}
    assert HF_TOKEN not in str(submission["request"])


def test_name_option_overrides_the_profile_name(profile, fake_api):
    mlp_validate("--finetune", "sft", "--name", "other")

    assert sent_submission(fake_api)["request"]["name"] == "other"


def test_validate_prints_the_resolved_request(profile):
    result = mlp_validate("--finetune", "sft")

    assert "resolved" in result.output


def test_validate_prints_each_rejection_with_its_path(profile, fake_api):
    detail = [{"loc": ["body", "request", "finetune", "base_model"], "msg": "is missing"}]
    fake_api.answers[api_paths.VALIDATE_PIPELINE] = (422, {"detail": detail})

    result = mlp_validate("--finetune", "sft")

    assert result.exit_code == 1
    assert "finetune.base_model: is missing" in result.output


def test_a_phase_missing_from_the_profile_is_an_error(profile, fake_api):
    result = mlp_validate("--finetune", "dpo")

    assert result.exit_code != 0
    assert "dpo" in result.output
    assert fake_api.received == []
