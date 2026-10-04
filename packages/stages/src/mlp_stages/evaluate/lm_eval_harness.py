import json
import math
import sys
from itertools import repeat
from pathlib import Path
from statistics import mean

from mlp_core import config
from mlp_stages.sandbox import run_in_sandbox


def main() -> None:
    """`python -m mlp_stages.evaluate.lm_eval_harness <lm-eval args>`: runs lm-eval."""
    score_code_in_sandbox()
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()


def score_code_in_sandbox() -> None:
    """lm-eval's coding tasks score with HF evaluate's `code_eval`; this one runs no code here."""
    import evaluate

    load = evaluate.load

    def load_metric(path: str, *args, **kwargs):
        return CodeEvalInSandbox() if path == "code_eval" else load(path, *args, **kwargs)

    evaluate.load = load_metric


class CodeEvalInSandbox:
    """`code_eval`'s `compute`, with each candidate program and its tests run in the Sandbox."""

    def compute(self, references: list[str], predictions: list[list[str]], k=(1, 10, 100), **_):
        """pass@k for each k no larger than every problem's candidate count, as `code_eval` does."""
        # 1. Every candidate of every problem in one batch, followed by its problem's tests.
        snippets = [
            {"code": f"{candidate}\n{tests}"}
            for candidates, tests in zip(predictions, references, strict=True)
            for candidate in candidates
        ]
        results = iter(run_in_sandbox(snippets))
        passed = [sum(next(results)["status"] == "ok" for _ in c) for c in predictions]

        # 2. The unbiased pass@k estimate, averaged over the problems.
        scores = {
            f"pass@{at}": mean(map(pass_at_k, map(len, predictions), passed, repeat(at)))
            for at in k
            if all(len(candidates) >= at for candidates in predictions)
        }
        return scores, {}


def pass_at_k(total: int, passed: int, k: int) -> float:
    """The chance that at least one of k candidates drawn from `total` passes."""
    if total - passed < k:
        return 1.0
    return 1 - math.comb(total - passed, k) / math.comb(total, k)


def download_command(task: str, output: Path) -> list[str]:
    """Scores one sample for lm-eval's built-in dummy model, which loads all the task needs."""
    return harness_run(task, "--model", "dummy", "--limit", "1")


def run_command(
    task: str, url: str, served_name: str, limit: int | None, output: Path
) -> list[str]:
    """lm-eval against the OpenAI-compatible server, tokenizing with the server's own tokenizer."""
    model_args = (
        f"model={served_name},base_url={url}/v1/completions,tokenizer_backend=remote,"
        f"num_concurrent={config.EVALUATE_CONCURRENT_REQUESTS},max_retries=3"
    )
    return harness_run(
        task,
        *("--model", "local-completions", "--model_args", model_args),
        *("--output_path", str(output)),
        *(["--limit", str(limit)] if limit else []),
    )


def harness_run(task: str, *arguments: str) -> list[str]:
    # lm-eval refuses tasks that run code unless told; only coding tasks are, sent to the Sandbox.
    coding = config.BENCHMARKS[f"lm_eval:{task}"]["category"] == "coding"
    unsafe = ["--confirm_run_unsafe_code"] if coding else []
    return [sys.executable, "-m", __name__, "run", "--tasks", task, *arguments, *unsafe]


def read_metrics(task: str, output: Path) -> dict[str, float]:
    """Every numeric metric of the task and its subtasks, as `<task>/<metric>`."""
    [results] = output.rglob("results_*.json")
    return {
        f"{name}/{metric.removesuffix(',none')}": value
        for name, metrics in json.loads(results.read_text())["results"].items()
        for metric, value in metrics.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


if __name__ == "__main__":
    main()
