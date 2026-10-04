import pytest
from typer.testing import CliRunner

from mlp_cli.main import app
from mlp_core import api_paths

STARTED = {"id": 7, "name": "smoketest-261002-120000", "kubeflow_run_url": "/pipeline/#/runs/1"}


def smoke_test(status, cases):
    return {"id": 7, "name": STARTED["name"], "status": status, "cases": cases}


@pytest.fixture
def started(logged_in, fake_api):
    fake_api.answers[api_paths.SMOKE_TEST_COMPLETE] = (202, STARTED)
    fake_api.answers[api_paths.SMOKE_TEST_CUSTOM] = (202, STARTED)


def mlp(*args):
    return CliRunner().invoke(app, list(args))


def test_smoke_test_runs_every_case_and_prints_each_result(started, fake_api):
    cases = {"fetch": "passed", "sft-lora-hf": "passed"}
    fake_api.answers[api_paths.PIPELINES] = (200, [smoke_test("succeeded", cases)])

    result = mlp("smoke-test")

    assert result.exit_code == 0, result.output
    assert fake_api.received[0][0] == api_paths.SMOKE_TEST_COMPLETE
    assert f"{fake_api.url}/pipeline/#/runs/1" in result.output
    assert "fetch: passed" in result.output
    assert "sft-lora-hf: passed" in result.output


def test_a_failed_case_fails_the_command(started, fake_api):
    cases = {"fetch": "failed", "sft-lora-hf": "passed"}
    fake_api.answers[api_paths.PIPELINES] = (200, [smoke_test("failed", cases)])

    result = mlp("smoke-test")

    assert result.exit_code == 1
    assert "fetch: failed" in result.output
    assert "sft-lora-hf: passed" in result.output


def test_a_custom_smoke_test_sends_only_the_named_cases(started, fake_api):
    fake_api.answers[api_paths.PIPELINES] = (200, [smoke_test("succeeded", {"fetch": "passed"})])

    result = mlp("smoke-test", "--phases", "sft", "--backends", "hf")

    assert result.exit_code == 0, result.output
    assert fake_api.received[0] == (
        api_paths.SMOKE_TEST_CUSTOM,
        {"finetune": {"phases": ["sft"], "backends": ["hf"]}},
    )


def test_a_custom_smoke_test_can_run_the_uploaded_model_case(started, fake_api):
    fake_api.answers[api_paths.PIPELINES] = (200, [smoke_test("succeeded", {"fetch": "passed"})])

    result = mlp("smoke-test", "--uploaded-model")

    assert result.exit_code == 0, result.output
    assert fake_api.received[0] == (api_paths.SMOKE_TEST_CUSTOM, {"uploaded_model": True})


def test_a_custom_smoke_test_can_run_the_sandbox_case(started, fake_api):
    fake_api.answers[api_paths.PIPELINES] = (200, [smoke_test("succeeded", {"fetch": "passed"})])

    result = mlp("smoke-test", "--sandbox")

    assert result.exit_code == 0, result.output
    assert fake_api.received[0] == (api_paths.SMOKE_TEST_CUSTOM, {"sandbox": True})


def test_a_custom_smoke_test_can_run_the_serving_cases(started, fake_api):
    fake_api.answers[api_paths.PIPELINES] = (200, [smoke_test("succeeded", {"fetch": "passed"})])

    result = mlp("smoke-test", "--serving")

    assert result.exit_code == 0, result.output
    assert fake_api.received[0] == (api_paths.SMOKE_TEST_CUSTOM, {"serving": True})


def test_a_custom_smoke_test_can_run_the_evaluate_cases(started, fake_api):
    fake_api.answers[api_paths.PIPELINES] = (200, [smoke_test("succeeded", {"fetch": "passed"})])

    result = mlp("smoke-test", "--evaluate")

    assert result.exit_code == 0, result.output
    assert fake_api.received[0] == (api_paths.SMOKE_TEST_CUSTOM, {"evaluate": True})


def test_a_smoke_test_already_running_is_reported(started, fake_api):
    fake_api.answers[api_paths.SMOKE_TEST_COMPLETE] = (409, {"detail": "still running"})

    result = mlp("smoke-test")

    assert result.exit_code == 1
    assert "still running" in result.output


def test_a_custom_smoke_test_can_run_the_distill_case(started, fake_api):
    fake_api.answers[api_paths.PIPELINES] = (200, [smoke_test("succeeded", {"fetch": "passed"})])

    result = mlp("smoke-test", "--distill")

    assert result.exit_code == 0, result.output
    assert fake_api.received[0] == (api_paths.SMOKE_TEST_CUSTOM, {"distill": True})
