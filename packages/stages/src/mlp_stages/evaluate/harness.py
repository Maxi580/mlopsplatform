import os
import re
import subprocess
import tempfile
from pathlib import Path

from mlp_core import config
from mlp_core.pipeline_request.references import benchmark_directory, split_benchmark_reference
from mlp_stages.evaluate import bfcl_harness, evalscope_harness, lm_eval_harness

# Each harness runs in its own process, which sends any code a benchmark generates to the Sandbox.
HARNESSES = {
    "lm_eval": lm_eval_harness,
    "evalscope": evalscope_harness,
    "bfcl": bfcl_harness,
}


def download_benchmark(benchmark: str) -> bool:
    """Whether the harness downloaded all the benchmark needs into its Model Cache directory."""
    # One sample scored for the harness's dummy model: besides the datasets, it fetches what a
    # task only loads while scoring, such as IFEval's sentence splitter.
    harness, task = split_benchmark_reference(benchmark)
    environment = benchmark_environment(benchmark)
    # NLTK downloads only into a directory that exists.
    Path(environment["NLTK_DATA"]).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as output:
        command = HARNESSES[harness].download_command(task, Path(output))
        return subprocess.run(command, env=environment).returncode == 0


def run_benchmark(
    benchmark: str, url: str, served_name: str, limit: int | None, scratch: Path
) -> dict[str, float] | None:
    """The benchmark's metrics by MLflow key, or None if its harness failed on it."""
    # 1. The harness against the OpenAI-compatible server, in its own process.
    harness, task = split_benchmark_reference(benchmark)
    output = scratch / harness / task
    command = HARNESSES[harness].run_command(task, url, served_name, limit, output)
    if subprocess.run(command, env=benchmark_environment(benchmark)).returncode:
        return None

    # 2. Its metrics, in keys MLflow accepts.
    metrics = HARNESSES[harness].read_metrics(task, output)
    return {mlflow_key(f"{harness}/{name}"): value for name, value in metrics.items()}


def benchmark_environment(benchmark: str) -> dict[str, str]:
    """The environment in which the harness keeps the benchmark's datasets in the Model Cache."""
    directory = benchmark_directory(Path(config.MODEL_CACHE_PATH), benchmark)
    return {
        **os.environ,
        "HF_DATASETS_CACHE": str(directory / "datasets"),
        "HF_HUB_CACHE": str(directory / "hub"),
        "EVALSCOPE_CACHE": str(directory / "evalscope"),
        "MODELSCOPE_CACHE": str(directory / "modelscope"),
        # IFEval's sentence splitter.
        "NLTK_DATA": str(directory / "nltk"),
    }


# MLflow keys allow neither the comma of lm-eval's `<metric>,<filter>` nor the @ of `pass@1`.
def mlflow_key(name: str) -> str:
    return re.sub(r"[^\w\-. /]", "/", name)
