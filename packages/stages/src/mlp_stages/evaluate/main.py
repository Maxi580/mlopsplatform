import os
import tempfile
from pathlib import Path

import mlflow

from mlp_core import config
from mlp_core.pipeline_request.references import model_reference
from mlp_core.pipeline_request.schema import PipelineRequest
from mlp_stages.evaluate.harness import mlflow_key, run_benchmark
from mlp_stages.evaluate.performance import measure_performance
from mlp_stages.served_model import served_model


def evaluate(pipeline_id: str, request: str, endpoint_url: str, gpus: str) -> None:
    """Runs each benchmark, and any performance run, and logs its metrics or NA into the Run."""
    # 1. The `evaluate` block of the resolved request the compiler passed in.
    resolved = PipelineRequest.model_validate_json(request)
    evaluate = resolved.evaluate
    model = evaluate.model
    if model in (config.FINETUNE_OUTPUT, config.QUANTIZE_OUTPUT):
        model = pipeline_output(resolved.name, pipeline_id, model)
    run_id = os.environ["MLFLOW_RUN_ID"]
    client = mlflow.MlflowClient()

    with tempfile.TemporaryDirectory() as scratch:
        # 2. The model behind an OpenAI-compatible URL: its Endpoint's, or a vLLM started here.
        served = served_model(
            model, evaluate.serving, resolved.name, endpoint_url, int(gpus), Path(scratch)
        )
        with served as (url, served_name, loaded):
            # Tells Runs apart that differ only by a `tool_parser` override.
            if evaluate.serving and evaluate.serving.tool_parser:
                client.set_tag(run_id, "tool_parser", evaluate.serving.tool_parser)

            # 3. Each benchmark in its own harness process, so a failing one fails alone.
            for benchmark in evaluate.benchmarks:
                metrics = run_benchmark(benchmark, url, served_name, evaluate.limit, Path(scratch))
                log_metrics(client, run_id, mlflow_key(benchmark), metrics)

            # 4. Serving performance, only when asked for, as it takes a while.
            if evaluate.performance:
                metrics = measure_performance(
                    evaluate.performance, url, served_name, loaded, Path(scratch)
                )
                log_metrics(client, run_id, config.PERFORMANCE_TOOL, metrics)


def log_metrics(client, run_id: str, name: str, metrics: dict[str, float] | None) -> None:
    """Logs the metrics into the Run, or tags `name` NA if its run failed."""
    if metrics is None:
        client.set_tag(run_id, name, "NA")
        print(f"{name} failed and is logged as NA; see its log above")
    for key, value in (metrics or {}).items():
        client.log_metric(run_id, key, value)


def pipeline_output(name: str, pipeline_id: str, stage_output: str) -> str:
    """The last Model Version the Pipeline registered as `@finetune` (a Phase's) or `@quantize`."""
    tag = "quantization" if stage_output == config.QUANTIZE_OUTPUT else "phase"
    versions = mlflow.MlflowClient().search_model_versions(f"name='{name}'")
    produced = [v for v in versions if v.tags.get("pipeline") == pipeline_id and tag in v.tags]
    if not produced:
        raise SystemExit(f"Pipeline {pipeline_id} registered no {stage_output} Model Version")
    return model_reference(name, max(int(version.version) for version in produced))
