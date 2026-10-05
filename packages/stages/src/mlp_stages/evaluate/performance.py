import json
import subprocess
import sys
from pathlib import Path

from mlp_core import config
from mlp_core.endpoint_spec import VllmModel
from mlp_core.pipeline_request.schema import Performance


def measure_performance(
    performance: Performance, url: str, served_name: str, loaded: VllmModel, scratch: Path
) -> dict[str, float] | None:
    """TTFT, inter-token latency and tokens/s by MLflow key; None if no request passed."""
    # 1. GuideLLM against the chat completions API, tokenizing its synthetic prompts as vLLM does.
    report = scratch / "guidellm.json"
    tokenizer = {"kind": "huggingface_auto", "model": loaded.path}
    if loaded.revision:
        tokenizer["load_kwargs"] = {"revision": loaded.revision}
    tokens = {
        "prompt_tokens": performance.prompt_tokens,
        "output_tokens": performance.output_tokens,
    }
    spec = {
        "backend": {"kind": "openai_http", "target": url, "model": served_name},
        "tokenizer": tokenizer,
        "data": [{"kind": "synthetic_text", **tokens}],
        "profile": {"kind": "concurrent", "streams": performance.concurrency},
        "constraints": [{"kind": "max_requests", "count": performance.requests}],
        "outputs": [{"kind": "json", "path": str(report)}],
    }
    scenario = scratch / "guidellm-scenario.json"
    scenario.write_text(json.dumps({"spec": spec}))
    run = ["run", "--scenario", str(scenario), "--disable-console-interactive"]
    if subprocess.run([sys.executable, "-m", "guidellm", *run]).returncode:
        return None

    # 2. Each metric's statistics over the successful requests; GuideLLM still exits cleanly when
    # every request failed, e.g. for a prompt longer than the model's context.
    [benchmark] = json.loads(report.read_text())["benchmarks"]
    if not benchmark["metrics"]["request_totals"]["successful"]:
        return None
    metrics = {}
    for metric in config.PERFORMANCE_METRICS:
        summary = benchmark["metrics"][metric]["successful"]
        values = {**summary, **summary["percentiles"]}
        for statistic in config.PERFORMANCE_STATISTICS:
            metrics[f"{config.PERFORMANCE_TOOL}/{metric}/{statistic}"] = values[statistic]
    return metrics
