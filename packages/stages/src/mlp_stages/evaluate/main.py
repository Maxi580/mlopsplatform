import os
import tempfile
from pathlib import Path

import mlflow

from mlp_core import config
from mlp_core.pipeline_request.references import model_reference
from mlp_core.pipeline_request.schema import PipelineRequest
from mlp_stages.evaluate.harness import mlflow_key, run_benchmark
from mlp_stages.served_model import served_model


def evaluate(pipeline_id: str, request: str, endpoint_url: str, gpus: str) -> None:
    """Runs each benchmark against the model and logs its metrics, or NA, into the step's Run."""
    # 1. The `evaluate` block of the resolved request the compiler passed in.
    resolved = PipelineRequest.model_validate_json(request)
    evaluate = resolved.evaluate
    model = evaluate.model
    if model == config.FINETUNE_OUTPUT:
        model = pipeline_output(resolved.name, pipeline_id)
    run_id = os.environ["MLFLOW_RUN_ID"]
    client = mlflow.MlflowClient()

    with tempfile.TemporaryDirectory() as scratch:
        # 2. The model behind an OpenAI-compatible URL: its Endpoint's, or a vLLM started here.
        served = served_model(
            model, evaluate.serving, resolved.name, endpoint_url, int(gpus), Path(scratch)
        )
        with served as (url, served_name):
            # 3. Each benchmark in its own lm-eval process, so a failing one fails alone.
            for benchmark in evaluate.benchmarks:
                metrics = run_benchmark(benchmark, url, served_name, evaluate.limit, Path(scratch))
                if metrics is None:
                    client.set_tag(run_id, mlflow_key(benchmark), "NA")
                    print(f"{benchmark} failed and is logged as NA; see lm-eval's log above")
                for key, value in (metrics or {}).items():
                    client.log_metric(run_id, key, value)


def pipeline_output(name: str, pipeline_id: str) -> str:
    """The `model:` Reference of the last Model Version the Pipeline registered."""
    versions = mlflow.MlflowClient().search_model_versions(f"name='{name}'")
    produced = [v for v in versions if v.tags.get("pipeline") == pipeline_id]
    if not produced:
        raise SystemExit(f"Pipeline {pipeline_id} registered no Model Version to evaluate")
    return model_reference(name, max(int(version.version) for version in produced))
