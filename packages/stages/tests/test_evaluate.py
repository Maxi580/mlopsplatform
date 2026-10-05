import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

mlflow = pytest.importorskip("mlflow", reason="needs the evaluate extra: uv sync --extra evaluate")

from mlp_core import config  # noqa: E402
from mlp_stages import served_model  # noqa: E402
from mlp_stages.evaluate import harness  # noqa: E402
from mlp_stages.main import main  # noqa: E402

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
COMMIT = "7ae557604adf67be50417f59c2c2f167def9a775"
BASE_MODEL = f"hf:{REPO}@{COMMIT}"
GSM8K_RESULTS = {
    "gsm8k": {"alias": "gsm8k", "exact_match,strict-match": 0.25, "exact_match_stderr,none": 0.1}
}
# GuideLLM's summary of each metric over the successful requests.
GUIDELLM_METRICS = {
    metric: {"successful": {"mean": mean, "median": mean - 1, "percentiles": {"p99": mean * 2}}}
    for metric, mean in [
        ("time_to_first_token_ms", 40.0),
        ("inter_token_latency_ms", 8.0),
        ("output_tokens_per_second", 900.0),
        ("requests_per_second", 3.0),
    ]
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


class FakeHarnesses:
    """Writes each task's results where its harness would; failing tasks exit with an error."""

    def __init__(self):
        self.commands = []
        self.environments = []
        self.results = {
            "gsm8k": GSM8K_RESULTS,
            "ifeval": {"ifeval": {"inst_level_strict_acc,none": 0.5}},
            "humaneval": {"humaneval": {"pass@1,create_test": 0.125}},
        }
        self.failing = set()
        self.scenarios = []

    def __call__(self, command, env=None):
        self.commands.append(command)
        self.environments.append(env)
        if "mlp_stages.evaluate.evalscope_harness" in command:
            return self.evalscope(json.loads(command[-1]))
        if "mlp_stages.evaluate.bfcl_harness" in command:
            return self.bfcl(json.loads(command[-1]))
        if "guidellm" in command:
            return self.guidellm(json.loads(Path(option(command, "--scenario")).read_text()))
        task = command[command.index("--tasks") + 1]
        if task in self.failing:
            return SimpleNamespace(returncode=1)
        output = Path(command[command.index("--output_path") + 1]) / "model"
        output.mkdir(parents=True)
        (output / "results_2026-10-04.json").write_text(json.dumps({"results": self.results[task]}))
        return SimpleNamespace(returncode=0)

    def evalscope(self, task_config):
        [task] = task_config["datasets"]
        reports = Path(task_config["work_dir"]) / "reports" / task_config["model"]
        reports.mkdir(parents=True)
        metrics = [
            {
                "identity": {"name": "accuracy", "aggregation": "mean", "dimensions": {}},
                "score": 0.5,
            },
            {
                "identity": {
                    "name": "accuracy",
                    "aggregation": "pass_at_k",
                    "dimensions": {"k": 1},
                },
                "score": 0.25,
            },
        ]
        (reports / f"{task}.json").write_text(json.dumps({"metrics": metrics}))
        return SimpleNamespace(returncode=0)

    def bfcl(self, settings):
        # BFCL's score file starts with the category's header line, followed by each failed entry.
        scores = Path(settings["output"]) / "score" / settings["served_name"] / "non_live"
        scores.mkdir(parents=True)
        header = {"accuracy": 0.75, "correct_count": 3, "total_count": 4}
        lines = [header, {"id": "simple_python_2", "valid": False}]
        score_file = scores / f"BFCL_v4_{settings['category']}_score.json"
        score_file.write_text("\n".join(map(json.dumps, lines)))
        return SimpleNamespace(returncode=0)

    def guidellm(self, scenario):
        self.scenarios.append(scenario)
        if "guidellm" in self.failing:
            return SimpleNamespace(returncode=1)
        [output] = scenario["spec"]["outputs"]
        successful = 0 if "every guidellm request" in self.failing else 10
        metrics = {**GUIDELLM_METRICS, "request_totals": {"successful": successful}}
        report = {"benchmarks": [{"metrics": metrics}]}
        Path(output["path"]).write_text(json.dumps(report))
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
    """Runs `mlp-stage evaluate`, with fakes for vLLM, the harnesses and MLflow."""
    fakes = SimpleNamespace(vllm=FakeVllm(), harnesses=FakeHarnesses(), mlflow=FakeMlflow())
    monkeypatch.setattr(config, "MODEL_CACHE_PATH", str(tmp_path))
    monkeypatch.setattr(served_model.subprocess, "Popen", fakes.vllm)
    monkeypatch.setattr(served_model.httpx, "get", lambda url: SimpleNamespace(is_success=True))
    monkeypatch.setattr(harness.subprocess, "run", fakes.harnesses)
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
    [lm_eval] = step.harnesses.commands
    assert option(lm_eval, "--model") == "local-completions"
    model_args = option(lm_eval, "--model_args")
    assert "model=qwen-sft,base_url=http://localhost:8000/v1/completions" in model_args
    assert "tokenizer_backend=remote" in model_args
    assert step.mlflow.metrics == {
        "lm_eval/gsm8k/exact_match/strict-match": 0.25,
        "lm_eval/gsm8k/exact_match_stderr": 0.1,
    }


def test_a_failing_benchmark_is_logged_as_na_and_the_rest_still_run(step):
    step.harnesses.failing.add("gsm8k")

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

    assert option(step.harnesses.commands[0], "--limit") == "5"


def test_an_endpoint_is_evaluated_through_its_service_without_a_vllm_of_its_own(step):
    step.run("endpoint:chat", endpoint_url="http://endpoint-chat.mlp.svc:8000", gpus="0")

    assert step.vllm.commands == []
    model_args = option(step.harnesses.commands[0], "--model_args")
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

    assert step.harnesses.commands == []


def test_a_coding_benchmark_runs_in_the_lm_eval_process_that_sends_its_code_to_the_sandbox(step):
    step.run(BASE_MODEL, benchmarks=["lm_eval:humaneval"])

    [lm_eval] = step.harnesses.commands
    assert lm_eval[:3] == [sys.executable, "-m", "mlp_stages.evaluate.lm_eval_harness"]
    assert "--confirm_run_unsafe_code" in lm_eval
    assert step.mlflow.metrics == {"lm_eval/humaneval/pass/1/create_test": 0.125}


def test_only_coding_benchmarks_may_run_generated_code(step):
    step.run(BASE_MODEL)

    assert "--confirm_run_unsafe_code" not in step.harnesses.commands[0]


def test_an_evalscope_benchmark_runs_against_chat_completions_from_the_model_cache(step, tmp_path):
    step.run(BASE_MODEL, benchmarks=["evalscope:mbpp_plus"], limit=5)

    [command] = step.harnesses.commands
    assert command[:3] == [sys.executable, "-m", "mlp_stages.evaluate.evalscope_harness"]
    task_config = json.loads(command[-1])
    assert task_config["model"] == "qwen-sft"
    assert task_config["eval_type"] == "openai_api"
    assert task_config["api_url"] == "http://localhost:8000/v1"
    assert task_config["limit"] == 5
    [environment] = step.harnesses.environments
    benchmark = tmp_path / "benchmarks" / "evalscope" / "mbpp_plus"
    assert environment["MODELSCOPE_CACHE"] == str(benchmark / "modelscope")
    assert environment["EVALSCOPE_CACHE"] == str(benchmark / "evalscope")
    assert step.mlflow.metrics == {
        "evalscope/mbpp_plus/accuracy/mean": 0.5,
        "evalscope/mbpp_plus/accuracy/pass_at_k/k1": 0.25,
    }


def test_with_performance_settings_guidellm_measures_the_models_own_vllm(step):
    performance = {"prompt_tokens": 64, "output_tokens": 32, "concurrency": 4, "requests": 20}

    step.run(BASE_MODEL, serving={"max_model_len": 2048}, performance=performance)

    [vllm] = step.vllm.commands
    assert option(vllm, "--max-model-len") == "2048"
    [scenario] = step.harnesses.scenarios
    spec = scenario["spec"]
    assert spec["backend"] == {
        "kind": "openai_http",
        "target": "http://localhost:8000",
        "model": "qwen-sft",
    }
    assert spec["tokenizer"]["model"] == REPO
    assert spec["tokenizer"]["load_kwargs"] == {"revision": COMMIT}
    assert spec["profile"] == {"kind": "concurrent", "streams": 4}
    assert spec["constraints"] == [{"kind": "max_requests", "count": 20}]
    [data] = spec["data"]
    assert (data["kind"], data["prompt_tokens"], data["output_tokens"]) == (
        "synthetic_text",
        64,
        32,
    )


def test_ttft_inter_token_latency_and_tokens_per_second_are_logged(step):
    step.run(BASE_MODEL, benchmarks=[], performance={})

    assert step.mlflow.metrics == {
        "guidellm/time_to_first_token_ms/mean": 40.0,
        "guidellm/time_to_first_token_ms/median": 39.0,
        "guidellm/time_to_first_token_ms/p99": 80.0,
        "guidellm/inter_token_latency_ms/mean": 8.0,
        "guidellm/inter_token_latency_ms/median": 7.0,
        "guidellm/inter_token_latency_ms/p99": 16.0,
        "guidellm/output_tokens_per_second/mean": 900.0,
        "guidellm/output_tokens_per_second/median": 899.0,
        "guidellm/output_tokens_per_second/p99": 1800.0,
    }


def test_without_performance_settings_no_performance_run_happens(step):
    step.run(BASE_MODEL)

    assert step.harnesses.scenarios == []
    assert not any(key.startswith("guidellm/") for key in step.mlflow.metrics)


def test_an_adapters_requests_are_tokenized_with_its_base_models_tokenizer(step):
    tags = {"weights": "adapter", "base_model": BASE_MODEL, "pipeline": "7"}
    step.mlflow.versions[("qwen-sft", 1)] = SimpleNamespace(version="1", tags=tags)

    step.run("@finetune", benchmarks=[], performance={})

    [scenario] = step.harnesses.scenarios
    assert scenario["spec"]["backend"]["model"] == "qwen-sft"
    assert scenario["spec"]["tokenizer"]["model"] == REPO


def test_a_failing_performance_run_is_logged_as_na_after_the_benchmarks(step):
    step.harnesses.failing.add("guidellm")

    step.run(BASE_MODEL, performance={})

    assert step.mlflow.tags == {"guidellm": "NA"}
    assert step.mlflow.metrics == {
        "lm_eval/gsm8k/exact_match/strict-match": 0.25,
        "lm_eval/gsm8k/exact_match_stderr": 0.1,
    }


def test_a_performance_run_without_a_successful_request_is_logged_as_na(step):
    step.harnesses.failing.add("every guidellm request")

    step.run(BASE_MODEL, benchmarks=[], performance={})

    assert step.mlflow.tags == {"guidellm": "NA"}
    assert step.mlflow.metrics == {}


def test_bfcl_calls_tools_through_a_vllm_serving_with_the_flags_of_serve(step):
    step.run(BASE_MODEL, benchmarks=["bfcl:simple_python"], serving={"tool_parser": "hermes"})

    [vllm] = step.vllm.commands
    assert "--enable-auto-tool-choice" in vllm
    assert option(vllm, "--tool-call-parser") == "hermes"
    [command] = step.harnesses.commands
    assert command[:3] == [config.BFCL_PYTHON, "-m", "mlp_stages.evaluate.bfcl_harness"]
    settings = json.loads(command[-1])
    assert settings["category"] == "simple_python"
    assert settings["url"] == "http://localhost:8000"
    assert settings["served_name"] == "qwen-sft"
    assert settings["limit"] is None
    assert step.mlflow.metrics == {"bfcl/simple_python/accuracy": 0.75}
    assert step.mlflow.tags == {"tool_parser": "hermes"}


def test_bfcl_runs_the_first_entries_of_its_category_up_to_the_limit(step):
    step.run(
        BASE_MODEL, benchmarks=["bfcl:simple_python"], serving={"tool_parser": "hermes"}, limit=5
    )

    assert json.loads(step.harnesses.commands[0][-1])["limit"] == 5
