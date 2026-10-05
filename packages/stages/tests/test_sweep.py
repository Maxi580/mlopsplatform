import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("optuna", reason="needs optuna: uv sync")
pytest.importorskip("datasets", reason="needs the evaluate extra: uv sync --extra evaluate")
mlflow = pytest.importorskip("mlflow", reason="needs the evaluate extra: uv sync --extra evaluate")

from mlp_stages import model_versions  # noqa: E402
from mlp_stages.main import main  # noqa: E402
from mlp_stages.sweep import main as sweep_main  # noqa: E402

BASE_MODEL = "hf:Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775"
BASE_MODEL_FILES = Path("/model-cache/hub/qwen")
ROWS = [{"prompt": f"question {number}", "completion": "answer"} for number in range(10)]
SETTINGS = {
    "learning_rate": 1e-4,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "max_length": 64,
}
LORA = {"r": 8, "lora_alpha": 16, "lora_dropout": 0.0, "target_modules": "all-linear"}


class FakeTrainer:
    """Logs an eval_loss that grows with the learning rate; a rank of 32 runs out of memory."""

    def __init__(self, phase, train_rows, eval_rows, output_directory):
        self.phase, self.train_rows, self.eval_rows = phase, train_rows, eval_rows
        self.output_directory = Path(output_directory)
        self.state = SimpleNamespace(log_history=[])

    def train(self):
        if self.phase.lora.r == 32:
            raise RuntimeError("CUDA out of memory")
        (self.output_directory / "checkpoint-1").mkdir()
        self.state.log_history.append({"loss": 2.0, "step": 1})

    def evaluate(self):
        self.state.log_history.append({"eval_loss": self.phase.settings.learning_rate * 1000})


class FakeMlflow:
    """The step's Run, and the nested Runs the Trials start under it."""

    def __init__(self):
        self.runs, self.logged = [], {}

    def __call__(self):
        return self

    def get_run(self, run_id):
        return SimpleNamespace(info=SimpleNamespace(experiment_id="5"))

    @contextmanager
    def start_run(self, parent_run_id, experiment_id, run_name):
        run = {"name": run_name, "parent": parent_run_id, "experiment": experiment_id}
        run.update(params={}, metrics={}, status="RUNNING")
        self.runs.append(run)
        try:
            yield run
        except Exception:
            run["status"] = "FAILED"
            raise
        run["status"] = "FINISHED"

    def log_params(self, params):
        self.runs[-1]["params"].update(params)

    def log_metric(self, *arguments):
        # mlflow.log_metric(key, value) in a Trial; MlflowClient().log_metric(run, key, value).
        *run_id, key, value = arguments
        target = self.logged if run_id else self.runs[-1]["metrics"]
        target[key] = value

    def log_param(self, run_id, key, value):
        self.logged[key] = value


@pytest.fixture
def step(monkeypatch, tmp_path):
    """Runs `mlp-stage sweep` with a fake backend, MLflow and API."""
    fakes = SimpleNamespace(mlflow=FakeMlflow(), trainers=[], reports=[])

    def build_trainer(model, phase, tokenizer, dataset, output, adapter, teacher, eval_dataset):
        assert (model, adapter, teacher) == (BASE_MODEL_FILES, None, None)
        fakes.trainers.append(FakeTrainer(phase, dataset, eval_dataset, output))
        return fakes.trainers[-1]

    backend = ModuleType("mlp_stages.finetune.backends.hf")
    backend.load_tokenizer = lambda path: "tokenizer"
    backend.build_trainer = build_trainer
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    # The trainer images have torch; this one may not.
    monkeypatch.setattr(sweep_main, "free_gpu_memory", lambda: None)
    monkeypatch.setattr(mlflow, "MlflowClient", fakes.mlflow)
    for name in ("start_run", "log_params", "log_metric"):
        monkeypatch.setattr(mlflow, name, getattr(fakes.mlflow, name))
    monkeypatch.setattr(
        model_versions, "snapshot_download", lambda repo, revision: BASE_MODEL_FILES
    )

    def download_dataset_version(reference, directory):
        path = directory / "rows.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in ROWS))
        return path

    monkeypatch.setattr(sweep_main, "download_dataset_version", download_dataset_version)
    monkeypatch.setattr(
        sweep_main, "call_step_route", lambda path, id, json: fakes.reports.append((path, id, json))
    )
    monkeypatch.setenv("MLFLOW_RUN_ID", "run-1")
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    output = tmp_path / "best_parameters"

    def run(**sweep):
        sweep = {
            "model": BASE_MODEL,
            "algorithm": "sft",
            "dataset": "dataset:chat@1",
            "settings": SETTINGS,
            "lora": LORA,
            "parameters": {"settings": {"learning_rate": {"values": [1e-4, 1e-5]}}},
            "objective": {"metric": "eval_loss", "goal": "minimize"},
            "trials": 2,
            "sampler": "grid",
            **sweep,
        }
        request = {"name": "qwen", "sweep": sweep}
        arguments = ["7", json.dumps(request), "", str(output)]
        monkeypatch.setattr(sys, "argv", ["mlp-stage", "sweep", *arguments])
        main()
        return json.loads(output.read_text())

    fakes.run = run
    return fakes


def test_each_trial_is_a_nested_run_and_the_best_parameters_are_reported(step):
    best = step.run()

    assert best == {"settings": {"learning_rate": 1e-5}, "lora": {}}
    assert [(run["parent"], run["experiment"], run["status"]) for run in step.mlflow.runs] == [
        ("run-1", "5", "FINISHED"),
        ("run-1", "5", "FINISHED"),
    ]
    tried = sorted(run["params"]["settings.learning_rate"] for run in step.mlflow.runs)
    assert tried == [1e-5, 1e-4]
    assert step.reports == [
        ("/pipelines/{id}/sweep", "7", {"parameters": best, "objective": pytest.approx(0.01)})
    ]
    assert step.mlflow.logged["best/eval_loss"] == pytest.approx(0.01)
    assert step.mlflow.logged["best/settings.learning_rate"] == 1e-5


def test_trials_hold_out_a_tenth_of_the_rows_and_keep_no_weights(step):
    step.run()

    for trainer in step.trainers:
        assert (len(trainer.train_rows), len(trainer.eval_rows)) == (9, 1)
        assert not trainer.output_directory.exists()


def test_a_failed_trial_is_logged_and_the_sweep_goes_on(step):
    best = step.run(parameters={"lora": {"r": {"values": [32, 16]}}})

    assert best == {"settings": {}, "lora": {"r": 16}}
    statuses = {run["params"]["lora.r"]: run["status"] for run in step.mlflow.runs}
    assert statuses == {32: "FAILED", 16: "FINISHED"}


def test_the_stage_fails_only_if_every_trial_failed(step):
    with pytest.raises(SystemExit, match="Every Trial failed"):
        step.run(parameters={"lora": {"r": {"values": [32]}}}, trials=1)

    assert step.reports == []


def test_ranges_sample_floats_and_integers_within_their_bounds(step):
    parameters = {
        "settings": {"learning_rate": {"min": 1e-6, "max": 1e-4, "scale": "log"}},
        "lora": {"r": {"min": 4, "max": 16}},
    }

    step.run(parameters=parameters, sampler="random", trials=3)

    for trainer in step.trainers:
        assert 1e-6 <= trainer.phase.settings.learning_rate <= 1e-4
        assert trainer.phase.lora.r in range(4, 17)
        assert trainer.phase.lora.lora_alpha == 16
