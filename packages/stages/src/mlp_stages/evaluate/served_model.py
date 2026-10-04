import json
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import httpx2 as httpx
import mlflow

from mlp_core import config
from mlp_core.endpoint_spec import ServingOptions, VllmModel, vllm_args
from mlp_core.pipeline_request.references import (
    model_reference,
    split_base_model_reference,
    split_endpoint_reference,
    split_model_reference,
)
from mlp_core.pipeline_request.schema import PipelineRequest


@contextmanager
def served_model(
    request: PipelineRequest, pipeline_id: str, endpoint_url: str, gpus: int, scratch: Path
) -> Iterator[tuple[str, str]]:
    """The URL and name the evaluated model is served under, until the block ends."""
    # 1. A running Endpoint, under its name.
    model = request.evaluate.model
    if endpoint_url:
        yield endpoint_url, split_endpoint_reference(model)
        return

    # 2. Otherwise a vLLM of its own, with the request's serving options, from the Model Cache.
    if model == config.FINETUNE_OUTPUT:
        model = pipeline_output(request.name, pipeline_id)
    options = request.evaluate.serving or ServingOptions()
    args = vllm_args(options, vllm_model(model, scratch), request.name, gpus)
    vllm = subprocess.Popen(["vllm", "serve", *args])
    try:
        wait_until_ready(vllm)
        yield config.LOCAL_VLLM_URL, request.name
    finally:
        vllm.terminate()
        vllm.wait()


def pipeline_output(name: str, pipeline_id: str) -> str:
    """The `model:` Reference of the last Model Version the Pipeline registered."""
    versions = mlflow.MlflowClient().search_model_versions(f"name='{name}'")
    produced = [v for v in versions if v.tags.get("pipeline") == pipeline_id]
    if not produced:
        raise SystemExit(f"Pipeline {pipeline_id} registered no Model Version to evaluate")
    return model_reference(name, max(int(version.version) for version in produced))


def vllm_model(reference: str, scratch: Path) -> VllmModel:
    """What vLLM loads for the `hf:` or `model:` Reference, downloading Model Versions first."""
    # 1. A Base Model, which fetch put into the Model Cache.
    if reference.startswith("hf:"):
        return VllmModel(*split_base_model_reference(reference))

    # 2. A Model Version's files: full weights, or an Adapter on the model under it.
    name, version = split_model_reference(reference)
    found = mlflow.MlflowClient().get_model_version(name, str(version))
    directory = mlflow.artifacts.download_artifacts(
        f"models:/{name}/{version}", dst_path=str(scratch / f"{name}-{version}")
    )
    if found.tags.get("weights") != "adapter":
        return VllmModel(directory)
    rank = json.loads((Path(directory) / "adapter_config.json").read_text())["r"]
    base = vllm_model(found.tags["base_model"], scratch)
    return replace(base, adapter_path=directory, adapter_rank=rank)


# vLLM exits when the model doesn't fit or load; its own log says why.
def wait_until_ready(vllm: subprocess.Popen) -> None:
    while vllm.poll() is None:
        try:
            if httpx.get(f"{config.LOCAL_VLLM_URL}/health").is_success:
                return
        except httpx.HTTPError:
            pass
        time.sleep(config.VLLM_READY_POLL_INTERVAL.total_seconds())
    raise SystemExit(f"vLLM exited with code {vllm.returncode} before serving the model")
