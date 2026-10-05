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
            "dpo": {"dataset": "dataset:pairs"},
        },
    },
    "evaluate": {"benchmarks": ["lm_eval:gsm8k"]},
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


def test_named_phases_become_the_requests_phases_in_the_order_named(profile, fake_api):
    mlp_validate("--finetune", "sft,dpo")

    phases = sent_submission(fake_api)["request"]["finetune"]["phases"]
    assert phases == [
        {"algorithm": "sft", "dataset": "dataset:chat"},
        {"algorithm": "dpo", "dataset": "dataset:pairs"},
    ]


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
    detail = [{"loc": ["finetune", "base_model"], "msg": "is missing"}]
    fake_api.answers[api_paths.VALIDATE_PIPELINE] = (422, {"detail": detail})

    result = mlp_validate("--finetune", "sft")

    assert result.exit_code == 1
    assert "finetune.base_model: is missing" in result.output


def test_a_phase_missing_from_the_profile_is_an_error(profile, fake_api):
    result = mlp_validate("--finetune", "kto")

    assert result.exit_code != 0
    assert "kto" in result.output
    assert fake_api.received == []


def test_serve_adds_the_profiles_serve_block(home, fake_api, platform_ca):
    write_profile(home, fake_api.url, platform_ca, **PROFILE, serve={"max_model_len": 4096})
    fake_api.answers[api_paths.VALIDATE_PIPELINE] = (200, {"request": {}})

    mlp_validate("--finetune", "sft", "--serve")

    assert sent_submission(fake_api)["request"]["serve"] == {"max_model_len": 4096}


def test_evaluate_adds_the_profiles_evaluate_block(home, fake_api, platform_ca):
    evaluate = {"model": "hf:Qwen/Qwen3-0.6B", "benchmarks": ["lm_eval:gsm8k"]}
    write_profile(home, fake_api.url, platform_ca, **{**PROFILE, "evaluate": evaluate})
    fake_api.answers[api_paths.VALIDATE_PIPELINE] = (200, {"request": {}})

    mlp_validate("--evaluate")

    assert sent_submission(fake_api)["request"] == {"name": PROFILE["name"], "evaluate": evaluate}


def test_serve_works_without_a_serve_block_in_the_profile(profile, fake_api):
    mlp_validate("--finetune", "sft", "--serve")

    assert sent_submission(fake_api)["request"]["serve"] == {}


def test_distill_adds_the_profiles_distill_block(home, fake_api, platform_ca):
    distill = {"dataset": "dataset:prompts", "teacher": "hf:Qwen/Qwen3-8B"}
    write_profile(home, fake_api.url, platform_ca, **{**PROFILE, "distill": distill})
    fake_api.answers[api_paths.VALIDATE_PIPELINE] = (200, {"request": {}})

    mlp_validate("--distill", "--finetune", "sft")

    request = sent_submission(fake_api)["request"]
    assert request["distill"] == distill
    assert request["finetune"]["phases"][0]["algorithm"] == "sft"


def test_speculate_adds_the_profiles_speculate_block(home, fake_api, platform_ca):
    speculate = {"speculator": "dflash", "dataset": "dataset:chat"}
    write_profile(home, fake_api.url, platform_ca, **{**PROFILE, "speculate": speculate})
    fake_api.answers[api_paths.VALIDATE_PIPELINE] = (200, {"request": {}})

    mlp_validate("--finetune", "sft", "--quantize", "--speculate")

    request = sent_submission(fake_api)["request"]
    assert list(request) == ["name", "finetune", "quantize", "speculate"]
    assert request["speculate"] == speculate


def test_sweep_adds_the_profiles_sweep_block_before_finetune(home, fake_api, platform_ca):
    sweep = {"algorithm": "sft", "trials": 4}
    write_profile(home, fake_api.url, platform_ca, **{**PROFILE, "sweep": sweep})
    fake_api.answers[api_paths.VALIDATE_PIPELINE] = (200, {"request": {}})

    mlp_validate("--finetune", "sft", "--sweep")

    request = sent_submission(fake_api)["request"]
    assert list(request) == ["name", "sweep", "finetune"]
    assert request["sweep"] == sweep


def test_quantize_adds_the_profiles_quantize_block_after_finetune(home, fake_api, platform_ca):
    quantize = {"scheme": "w4a16-gptq", "calibration": {}}
    write_profile(home, fake_api.url, platform_ca, **{**PROFILE, "quantize": quantize})
    fake_api.answers[api_paths.VALIDATE_PIPELINE] = (200, {"request": {}})

    mlp_validate("--finetune", "sft", "--quantize")

    request = sent_submission(fake_api)["request"]
    assert list(request) == ["name", "finetune", "quantize"]
    assert request["quantize"] == quantize
