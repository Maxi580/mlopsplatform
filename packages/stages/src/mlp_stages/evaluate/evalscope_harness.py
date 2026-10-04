import json
import sys
from pathlib import Path

from mlp_core import config
from mlp_stages.sandbox import run_in_sandbox


def main() -> None:
    """`python -m mlp_stages.evaluate.evalscope_harness <TaskConfig JSON>`: runs EvalScope."""
    score_code_in_sandbox()
    from evalscope import TaskConfig, run_task

    run_task(TaskConfig(**json.loads(sys.argv[1])))


def score_code_in_sandbox() -> None:
    """Every EvalScope coding task runs its code through this mixin: in the Sandbox, never here."""
    from evalscope.api.mixin import CodeExecutionSandboxMixin

    CodeExecutionSandboxMixin.use_sandbox = property(lambda benchmark: True)
    CodeExecutionSandboxMixin.execute_code_in_sandbox = execute_code_in_sandbox


def execute_code_in_sandbox(benchmark, code, timeout: int = 60, language: str = "python") -> dict:
    """The result as EvalScope's tasks read it: `status` is `success` once the code ran cleanly."""
    # The Sandbox runs only Python, within its own timeout.
    if language != "python" or not isinstance(code, str):
        return {"status": "error", "error": "The Sandbox runs a single Python script only"}
    [result] = run_in_sandbox([{"code": code}])
    status = "success" if result["status"] == "ok" else result["status"]
    return {"status": status, "output": result["stdout"], "error": result["stderr"]}


def download_command(task: str, output: Path) -> list[str]:
    """Scores one sample for EvalScope's mock model, which loads and keeps the whole dataset."""
    return harness_run({"model": "mock", "eval_type": "mock_llm", "limit": 1}, task, output)


def run_command(
    task: str, url: str, served_name: str, limit: int | None, output: Path
) -> list[str]:
    """EvalScope against the OpenAI-compatible server's chat completions."""
    model_settings = {
        "model": served_name,
        "eval_type": "openai_api",
        "api_url": f"{url}/v1",
        "eval_batch_size": config.EVALUATE_CONCURRENT_REQUESTS,
        "limit": limit,
    }
    return harness_run(model_settings, task, output)


def harness_run(model_settings: dict, task: str, output: Path) -> list[str]:
    # Its datasets come from MODELSCOPE_CACHE, which the benchmark's environment points at.
    task_config = {**model_settings, "datasets": [task], "work_dir": str(output)}
    return [sys.executable, "-m", __name__, json.dumps({**task_config, "no_timestamp": True})]


def read_metrics(task: str, output: Path) -> dict[str, float]:
    """Each overall metric of the task's report, as `<task>/<metric>/<aggregation>[/<k><v>]`."""
    [report] = (output / "reports").rglob(f"{task}.json")
    metrics = {}
    # A metric no sample could be scored on has no score.
    for metric in json.loads(report.read_text())["metrics"]:
        if metric["score"] is None:
            continue
        identity = metric["identity"]
        dimensions = [f"{key}{value}" for key, value in identity["dimensions"].items()]
        key = "/".join([task, identity["name"], identity["aggregation"], *dimensions])
        metrics[key] = metric["score"]
    return metrics


if __name__ == "__main__":
    main()
