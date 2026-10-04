import json
import os
import re
import subprocess
import sys
from pathlib import Path

from mlp_core import config
from mlp_core.pipeline_request.references import benchmark_directory, split_benchmark_reference


def download_benchmark(benchmark: str) -> bool:
    """Whether lm-eval downloaded all the benchmark needs into its Model Cache directory."""
    # One sample scored for lm-eval's built-in dummy model: besides the datasets, it fetches
    # what a task only loads while scoring, such as IFEval's sentence splitter.
    _, task = split_benchmark_reference(benchmark)
    environment = benchmark_environment(benchmark)
    # NLTK downloads only into a directory that exists.
    Path(environment["NLTK_DATA"]).mkdir(parents=True, exist_ok=True)
    command = lm_eval_run("--model", "dummy", "--tasks", task, "--limit", "1")
    return subprocess.run(command, env=environment).returncode == 0


def run_benchmark(
    benchmark: str, url: str, served_name: str, limit: int | None, scratch: Path
) -> dict[str, float] | None:
    """The benchmark's metrics by MLflow key, or None if lm-eval failed on it."""
    # 1. lm-eval against the OpenAI-compatible server, tokenizing with the server's own tokenizer.
    harness, task = split_benchmark_reference(benchmark)
    output = scratch / harness / task
    model_args = (
        f"model={served_name},base_url={url}/v1/completions,tokenizer_backend=remote,"
        f"num_concurrent={config.EVALUATE_CONCURRENT_REQUESTS},max_retries=3"
    )
    command = lm_eval_run(
        *("--model", "local-completions", "--model_args", model_args),
        *("--tasks", task, "--output_path", str(output)),
        *(["--limit", str(limit)] if limit else []),
    )
    if subprocess.run(command, env=benchmark_environment(benchmark)).returncode:
        return None

    # 2. Every numeric metric of the task and its subtasks, in keys MLflow accepts.
    [results] = output.rglob("results_*.json")
    return {
        mlflow_key(f"{harness}/{name}/{metric.removesuffix(',none')}"): value
        for name, metrics in json.loads(results.read_text())["results"].items()
        for metric, value in metrics.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


def lm_eval_run(*arguments: str) -> list[str]:
    return [sys.executable, "-m", "lm_eval", "run", *arguments]


def benchmark_environment(benchmark: str) -> dict[str, str]:
    """The environment in which lm-eval keeps the benchmark's datasets in the Model Cache."""
    directory = benchmark_directory(Path(config.MODEL_CACHE_PATH), benchmark)
    return {
        **os.environ,
        "HF_DATASETS_CACHE": str(directory / "datasets"),
        "HF_HUB_CACHE": str(directory / "hub"),
        # IFEval's sentence splitter.
        "NLTK_DATA": str(directory / "nltk"),
    }


# lm-eval names a metric `<metric>,<filter>`; MLflow keys allow no commas.
def mlflow_key(name: str) -> str:
    return re.sub(r"[^\w\-. /]", "/", name)
