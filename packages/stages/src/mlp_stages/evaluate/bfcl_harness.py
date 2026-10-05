import json
import os
import sys
from pathlib import Path

from mlp_core import config


def main() -> None:
    """`<BFCL python> -m mlp_stages.evaluate.bfcl_harness <settings JSON>`: runs a BFCL category."""
    # 1. Where BFCL writes its results and scores, and the server it calls; both are read on import.
    settings = json.loads(sys.argv[1])
    output = Path(settings["output"]).resolve()
    os.environ["BFCL_PROJECT_ROOT"] = str(output)
    os.environ["OPENAI_BASE_URL"] = f"{settings['url']}/v1"
    os.environ["OPENAI_API_KEY"] = "unused"
    from bfcl_eval.__main__ import cli
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
    from bfcl_eval.utils import load_dataset_entry

    # 2. The served model behind BFCL's OpenAI handler, which sends real `tools` to chat
    # completions, so the score covers the chat template and tool parser too (#18). That
    # handler names functions with `_` for `.`, which the checker must know.
    name = settings["served_name"]
    MODEL_CONFIG_MAPPING[name] = ModelConfig(
        model_name=name,
        display_name=name,
        url="",
        org="",
        license="",
        model_handler=OpenAICompletionsHandler,
        underscore_to_dot=True,
    )

    # 3. The category's replies, or those to its first `limit` entries, then their scores.
    category = settings["category"]
    threads = str(config.EVALUATE_CONCURRENT_REQUESTS)
    generate = ["generate", "--model", name, "--test-category", category, "--num-threads", threads]
    evaluate = ["evaluate", "--model", name, "--test-category", category]
    if settings["limit"]:
        ids = [entry["id"] for entry in load_dataset_entry(category)[: settings["limit"]]]
        (output / "test_case_ids_to_generate.json").write_text(json.dumps({category: ids}))
        generate.append("--run-ids")
        evaluate.append("--partial-eval")
    cli(generate, standalone_mode=False)
    cli(evaluate, standalone_mode=False)


# BFCL ships its datasets inside its package, so there is nothing to download.
def download_command(task: str, output: Path) -> list[str]:
    return [config.BFCL_PYTHON, "-c", "import bfcl_eval"]


def run_command(
    task: str, url: str, served_name: str, limit: int | None, output: Path
) -> list[str]:
    settings = {
        "category": task,
        "url": url,
        "served_name": served_name,
        "limit": limit,
        "output": str(output),
    }
    return [config.BFCL_PYTHON, "-m", __name__, json.dumps(settings)]


def read_metrics(task: str, output: Path) -> dict[str, float]:
    """The category's accuracy, as `<task>/accuracy`, from the header line of its score file."""
    [scores] = (output / "score").rglob(f"BFCL_v4_{task}_score.json")
    header = json.loads(scores.read_text().splitlines()[0])
    return {f"{task}/accuracy": header["accuracy"]}


if __name__ == "__main__":
    main()
