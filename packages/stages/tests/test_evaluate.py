import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

mlflow = pytest.importorskip("mlflow", reason="needs the evaluate extra: uv sync --extra evaluate")

from mlp_core import config  # noqa: E402
from mlp_stages.evaluate import harness, served_model  # noqa: E402
from mlp_stages.main import main  # noqa: E402

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
COMMIT = "7ae557604adf67be50417f59c2c2f167def9a775"
BASE_MODEL = f"hf:{REPO}@{COMMIT}"
GSM8K_RESULTS = {
    "gsm8k": {"alias": "gsm8k", "exact_match,strict-match": 0.25, "exact_match_stderr,none": 0.1}
}


class FakeVllm:
    """A `vllm serve` process that serves until terminated, unless told to exit at once."""

    def __init__(self):
        self.commands = []
        self.exits_at_once = False
        self.terminated = False

    def __call__(self, command):
        self.commands.append(command)
        self.returncode = 1 if self.exits_at_once else None
        return self

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self):
        return self.returncode


class FakeLmEval:
    """Writes each task's results where lm-eval would; failing tasks exit with an error."""

    def __init__(self):
        self.commands = []
        self.results = {
            "gsm8k": GSM8K_RESULTS,
            "ifeval": {"ifeval": {"inst_level_strict_acc,none": 0.5}},
        }
        self.failing = set()

    def __call__(self, command, env):
        self.commands.append(command)
        task = command[command.index("--tasks") + 1]
        if task in self.failing:
            return SimpleNamespace(returncode=1)
        output = Path(command[command.index("--output_path") + 1]) / "model"
        output.mkdir(parents=True)
        (output / "results_2026-10-04.json").write_text(json.dumps({"results": self.results[task]}))
        return SimpleNamespace(returncode=0)


class FakeMlflow:
    """The step's Run and the registry's Model Versions, whose files land where asked."""

    def __init__(self):
        self.metrics = {}
        self.tags = {}
        self.versions = {}

    def __call__(self):
        return self

    def log_metric(self, run_id, key, value):
        self.metrics[key] = value

    def set_tag(self, run_id, key, value):
        self.tags[key] = value

    def search_model_versions(self, filter_string):
        return [v for (name, _), v in self.versions.items() if f"name='{name}'" == filter_string]

    def get_model_version(self, name, version):
        return self.versions[(name, int(version))]

    def download_artifacts(self, uri, dst_path):
        Path(dst_path).mkdir(parents=True)
        (Path(dst_path) / "adapter_config.json").write_text(json.dumps({"r": 16}))
        return dst_path


@pytest.fixture
def step(monkeypatch, tmp_path):
    """Runs `mlp-stage evaluate`, with fakes for vLLM, lm-eval and MLflow."""
    fakes = SimpleNamespace(vllm=FakeVllm(), lm_eval=FakeLmEval(), mlflow=FakeMlflow())
    monkeypatch.setattr(config, "MODEL_CACHE_PATH", str(tmp_path))
    monkeypatch.setattr(served_model.subprocess, "Popen", fakes.vllm)
    monkeypatch.setattr(served_model.httpx, "get", lambda url: SimpleNamespace(is_success=True))
    monkeypatch.setattr(harness.subprocess, "run", fakes.lm_eval)
    monkeypatch.setattr(mlflow, "MlflowClient", fakes.mlflow)
    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", fakes.mlflow.download_artifacts)
    monkeypatch.setenv("MLFLOW_RUN_ID", "run-1")
    # main() wraps stdout and stderr; monkeypatch puts the originals back.
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)

    def run(model, benchmarks=("lm_eval:gsm8k",), endpoint_url="", gpus="1", **evaluate):
        evaluate = {"model": model, "benchmarks": list(benchmarks), **evaluate}
        request = {"name": "qwen-sft", "evaluate": evaluate}
        monkeypatch.setattr(sys, "argv", ["mlp-stage", "evaluate", "7", json.dumps(request)])
        sys.argv += [endpoint_url, gpus]
        main()

    fakes.run = run
    return fakes


def option(command, name):
    return command[command.index(name) + 1]


def test_a_base_model_is_served_by_vllm_and_each_metric_logged(step):
    step.run(BASE_MODEL)

    [vllm] = step.vllm.commands
    assert vllm[:3] == ["vllm", "serve", REPO]
    assert option(vllm, "--revision") == COMMIT
    assert option(vllm, "--tensor-parallel-size") == "1"
    assert "--enable-tokenizer-info-endpoint" in vllm
    assert step.vllm.terminated
    [lm_eval] = step.lm_eval.commands
    assert option(lm_eval, "--model") == "local-completions"
    model_args = option(lm_eval, "--model_args")
    assert "model=qwen-sft,base_url=http://localhost:8000/v1/completions" in model_args
    assert "tokenizer_backend=remote" in model_args
    assert step.mlflow.metrics == {
        "lm_eval/gsm8k/exact_match/strict-match": 0.25,
        "lm_eval/gsm8k/exact_match_stderr": 0.1,
    }


def test_a_failing_benchmark_is_logged_as_na_and_the_rest_still_run(step):
    step.lm_eval.failing.add("gsm8k")

    step.run(BASE_MODEL, benchmarks=["lm_eval:gsm8k", "lm_eval:ifeval"])

    assert step.mlflow.tags == {"lm_eval/gsm8k": "NA"}
    assert step.mlflow.metrics == {"lm_eval/ifeval/inst_level_strict_acc": 0.5}


def test_vllm_serves_with_the_requests_serving_options(step):
    step.run(BASE_MODEL, serving={"max_model_len": 2048, "dtype": "bfloat16"})

    [vllm] = step.vllm.commands
    assert option(vllm, "--max-model-len") == "2048"
    assert option(vllm, "--dtype") == "bfloat16"


def test_the_limit_caps_the_samples_per_task(step):
    step.run(BASE_MODEL, limit=5)

    assert option(step.lm_eval.commands[0], "--limit") == "5"


def test_an_endpoint_is_evaluated_through_its_service_without_a_vllm_of_its_own(step):
    step.run("endpoint:chat", endpoint_url="http://endpoint-chat.mlp.svc:8000", gpus="0")

    assert step.vllm.commands == []
    model_args = option(step.lm_eval.commands[0], "--model_args")
    assert "model=chat,base_url=http://endpoint-chat.mlp.svc:8000/v1/completions" in model_args


def test_the_output_of_finetune_is_the_pipelines_last_model_version_on_its_base(step):
    tags = {"weights": "adapter", "base_model": BASE_MODEL, "pipeline": "7"}
    for version in (1, 2):
        step.mlflow.versions[("qwen-sft", version)] = SimpleNamespace(
            version=str(version), tags=tags
        )
    step.mlflow.versions[("qwen-sft", 3)] = SimpleNamespace(version="3", tags={"pipeline": "8"})

    step.run("@finetune")

    [vllm] = step.vllm.commands
    assert vllm[2] == REPO
    lora = option(vllm, "--lora-modules")
    assert lora.startswith("qwen-sft=") and lora.endswith("qwen-sft-2")
    assert option(vllm, "--max-lora-rank") == "16"


def test_a_vllm_that_exits_before_serving_fails_the_step(step):
    step.vllm.exits_at_once = True

    with pytest.raises(SystemExit, match="vLLM exited with code 1"):
        step.run(BASE_MODEL)

    assert step.lm_eval.commands == []
