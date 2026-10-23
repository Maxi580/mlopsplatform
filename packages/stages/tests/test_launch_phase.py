import json
import signal
import sys
from types import ModuleType, SimpleNamespace

import pytest

from mlp_stages.finetune import trainjob
from mlp_stages.main import main

TRAINJOB = {
    "metadata": {"generateName": "pipeline-7-phase-0-", "labels": {"mlp/pipeline-id": "7"}},
    "spec": {"runtimeRef": {"name": "mlp-finetune"}, "trainer": {"image": "mlp-trainer-hf:test"}},
}
PHASE_ARGUMENTS = ["7", "0", '{"name": "qwen-sft"}', "", "", "", "30", "", ""]


class FakeTrainJobs:
    """TrainJobs that print `lines` and end as `outcome`: succeeded, failed or cancelled."""

    def __init__(self, outcome: str, lines: list[str]):
        self.outcome, self.lines = outcome, lines
        self.created, self.deleted = [], []

    def __call__(self):
        return self

    def create(self, spec: dict) -> str:
        self.created.append(spec)
        return "pipeline-7-phase-0-x1"

    def logs(self, name: str):
        yield from self.lines
        # KFP terminates the step, e.g. when the Pipeline is cancelled.
        if self.outcome == "cancelled":
            signal.raise_signal(signal.SIGTERM)

    def succeeded(self, name: str) -> bool:
        return self.outcome == "succeeded"

    def delete(self, name: str) -> None:
        self.deleted.append(name)


@pytest.fixture
def launch(monkeypatch, tmp_path):
    """Runs `mlp-stage launch-phase` against fake TrainJobs and an MLflow holding one version."""
    mlflow = ModuleType("mlflow")
    registered = SimpleNamespace(name="qwen-sft", version="3")
    searches = []

    def search_model_versions(filter_string):
        searches.append(filter_string)
        return [registered]

    mlflow.MlflowClient = lambda: SimpleNamespace(search_model_versions=search_model_versions)
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    monkeypatch.setenv("MLFLOW_RUN_ID", "run-1")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.test")
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    output = tmp_path / "model_version"
    fakes = SimpleNamespace(output=output, searches=searches, jobs=None)

    def run(outcome="succeeded", lines=("step 1",), stop_after_checkpoint=""):
        fakes.jobs = FakeTrainJobs(outcome, list(lines))
        monkeypatch.setattr(trainjob, "TrainJobs", fakes.jobs)
        arguments = [json.dumps(TRAINJOB), *PHASE_ARGUMENTS[:-1], stop_after_checkpoint]
        monkeypatch.setattr(sys, "argv", ["mlp-stage", "launch-phase", *arguments, str(output)])
        main()

    fakes.run = run
    # The launcher's SIGTERM handler is only for its run.
    previous = signal.getsignal(signal.SIGTERM)
    yield fakes
    signal.signal(signal.SIGTERM, previous)


def test_a_phase_runs_as_its_trainjob_in_the_steps_mlflow_run(launch):
    launch.run()

    [created] = launch.jobs.created
    trainer = created["spec"]["trainer"]
    assert trainer["image"] == "mlp-trainer-hf:test"
    # The Phase writes its own output nowhere; the launcher writes the step's.
    assert trainer["args"][:-1] == PHASE_ARGUMENTS
    assert {"name": "MLFLOW_RUN_ID", "value": "run-1"} in trainer["env"]
    assert {"name": "MLFLOW_TRACKING_URI", "value": "http://mlflow.test"} in trainer["env"]
    assert created["metadata"]["labels"] == {"mlp/pipeline-id": "7"}


def test_the_step_streams_the_phases_log_and_outputs_the_model_version_it_registered(
    launch, capsys
):
    launch.run(lines=["step 1", "step 2"])

    assert "step 1\nstep 2\n" in capsys.readouterr().out
    assert launch.searches == ["run_id = 'run-1'"]
    assert launch.output.read_text() == "model:qwen-sft@3"


def test_a_failed_trainjob_fails_the_step_with_the_main_processs_last_lines(launch):
    lines = [f"line {number}" for number in range(30)] + ["Out of GPU memory: CUDA"]

    with pytest.raises(SystemExit) as failure:
        launch.run(outcome="failed", lines=lines)

    message = str(failure.value)
    assert message.startswith("The Phase's TrainJob pipeline-7-phase-0-x1 failed")
    assert message.endswith("line 29\nOut of GPU memory: CUDA")
    assert "line 0\n" not in message
    assert not launch.output.exists()


def test_a_cancelled_step_deletes_its_trainjob(launch):
    with pytest.raises(SystemExit):
        launch.run(outcome="cancelled")

    assert launch.jobs.deleted == ["pipeline-7-phase-0-x1"]
    assert not launch.output.exists()


def test_an_interrupted_phase_outputs_no_model_version(launch):
    launch.run(stop_after_checkpoint="true")

    assert launch.searches == []
    assert launch.output.read_text() == ""
