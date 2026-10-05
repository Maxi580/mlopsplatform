import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

mlflow = pytest.importorskip("mlflow", reason="needs the evaluate extra: uv sync --extra evaluate")

from mlp_stages import model_versions  # noqa: E402
from mlp_stages.main import main  # noqa: E402
from mlp_stages.speculate import main as speculate_main  # noqa: E402

from .test_quantize import BASE_MODEL, FakeMlflow  # noqa: E402

ROWS = [
    {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]},
    {
        "prompt": [{"role": "user", "content": "Weather?"}],
        "completion": [{"role": "assistant", "content": "Sunny."}],
    },
]


class FakeRun(FakeMlflow):
    """The registry and the step's Run, which keeps its params and the files logged into it."""

    def __init__(self):
        super().__init__()
        self.params = {}

    def log_param(self, run_id, key, value):
        self.params[key] = value

    def log_artifacts(self, run_id, directory, path):
        self.logged = sorted(file.name for file in Path(directory).iterdir())
        self.speculator_config = json.loads((Path(directory) / "config.json").read_text())


@pytest.fixture
def step(monkeypatch, tmp_path):
    """Runs `mlp-stage speculate`, with fakes for vLLM, `speculators` and MLflow."""
    fakes = FakeRun()
    fakes.commands, fakes.vllm = [], []
    verifier_files = tmp_path / "verifier"
    verifier_files.mkdir()
    (verifier_files / "config.json").write_text(json.dumps({"num_hidden_layers": 24}))
    monkeypatch.setattr(mlflow, "MlflowClient", fakes)

    def download_artifacts(uri, dst_path):
        (Path(fakes.download_artifacts(uri, dst_path)) / "config.json").write_text(
            json.dumps({"num_hidden_layers": 24})
        )
        return dst_path

    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", download_artifacts)
    monkeypatch.setattr(model_versions, "snapshot_download", lambda repo, revision: verifier_files)

    @contextmanager
    def running_vllm(args):
        fakes.vllm.append(args)
        yield

    # prepare-data reads the conversations; training writes each epoch, and its best one again as
    # `checkpoint_best`.
    def run(command, check, cwd):
        fakes.commands.append(command)
        if "prepare-data" in command:
            data = Path(command[command.index("--data") + 1]).read_text().splitlines()
            fakes.conversations = [json.loads(line)["conversations"] for line in data]
        if "speculators.train" in command:
            save = Path(command[command.index("--save-path") + 1])
            for epoch in ("epoch_0", "epoch_1", "checkpoint_best"):
                (save / epoch).mkdir(parents=True)
                config = {"best": epoch == "checkpoint_best", "speculators_config": {}}
                (save / epoch / "config.json").write_text(json.dumps(config))
                (save / epoch / "optimizer_state.pt").write_text("state")

    def download_dataset_version(reference, directory):
        fakes.dataset = reference
        path = directory / "rows.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in ROWS))
        return path

    monkeypatch.setattr(speculate_main, "running_vllm", running_vllm)
    monkeypatch.setattr(speculate_main.subprocess, "run", run)
    monkeypatch.setattr(speculate_main, "download_dataset_version", download_dataset_version)
    monkeypatch.setenv("MLFLOW_RUN_ID", "run-1")
    # main() wraps stdout and stderr; monkeypatch puts the originals back.
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    output = tmp_path / "model_version"

    def run_step(model=BASE_MODEL, handed=("", "", ""), **speculate):
        speculate = {"model": model, "dataset": "dataset:chat@1", **speculate}
        request = json.dumps({"name": "qwen-sft", "speculate": speculate})
        arguments = ["7", request, *handed, "2", "4", str(output)]
        monkeypatch.setattr(sys, "argv", ["mlp-stage", "speculate", *arguments])
        main()
        return output.read_text()

    fakes.run = run_step
    return fakes


def command(step, name: str) -> list[str]:
    return next(c for c in step.commands if name in c)


def flag(arguments: list[str], name: str) -> str:
    return arguments[arguments.index(name) + 1]


def test_the_best_epoch_registers_as_a_speculator_for_the_verifier(step):
    registered = step.run(speculator="dflash", settings={"samples": 50})

    assert registered == "model:qwen-sft-speculator@3"
    [(name, tags)] = step.registered
    assert name == "qwen-sft-speculator"
    assert tags == {
        "weights": "speculator",
        "speculator": "dflash",
        "verifier": BASE_MODEL,
        "pipeline": "7",
    }
    assert step.speculator_config["best"]
    assert step.speculator_config["speculators_config"]["verifier"]["name_or_path"] == BASE_MODEL
    assert step.logged == ["config.json", "pipeline_request.json"]
    assert step.params["samples"] == 50


def test_vllm_hands_out_four_verifier_layers_until_training_starts(step):
    step.run()

    [args] = step.vllm
    assert flag(args, "--tensor-parallel-size") == "2"
    assert "--no-enable-prefix-caching" in args
    extraction = json.loads(flag(args, "--speculative-config"))
    assert extraction["method"] == "extract_hidden_states"
    layers = extraction["draft_model_config"]["hf_config"]["eagle_aux_hidden_state_layer_ids"]
    assert layers == [3, 12, 21, 24]
    assert json.loads(flag(args, "--kv-transfer-config"))["kv_role"] == "kv_producer"
    assert [c[1] for c in step.commands[:2]] == ["prepare-data", "generate-offline-data"]
    train = command(step, "speculators.train")
    assert (flag(train, "--speculator-type"), flag(train, "--num-workers")) == ("eagle3", "4")
    assert flag(train, "--on-missing") == "raise"


def test_conversations_come_from_messages_rows_and_distilled_replies(step):
    step.run(dataset="@distill", handed=("", "", "dataset:qwen-sft@2"))

    assert step.dataset == "dataset:qwen-sft@2"
    assert step.conversations == [ROWS[0]["messages"], ROWS[1]["prompt"] + ROWS[1]["completion"]]


def test_the_output_of_quantize_is_the_verifier_kfp_hands_over(step):
    step.versions[("qwen-sft", 2)] = {"weights": "full"}

    step.run("@quantize", handed=("model:qwen-sft@1", "model:qwen-sft@2", ""))

    assert "models:/qwen-sft/2" in step.downloads
    [(_, tags)] = step.registered
    assert tags["verifier"] == "model:qwen-sft@2"
