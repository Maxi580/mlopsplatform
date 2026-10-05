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
    split_base_model_reference,
    split_endpoint_reference,
    split_model_reference,
)


@contextmanager
def served_model(
    model: str,
    options: ServingOptions | None,
    served_name: str,
    endpoint_url: str,
    gpus: int,
    scratch: Path,
) -> Iterator[tuple[str, str, VllmModel | None]]:
    """The URL and name it is served under, and what vLLM loads (None for an Endpoint)."""
    # 1. A running Endpoint, under its name.
    if endpoint_url:
        yield endpoint_url, split_endpoint_reference(model), None
        return

    # 2. Otherwise a vLLM of its own, with the serving options, from the Model Cache.
    loaded = vllm_model(model, scratch)
    with running_vllm(vllm_args(options or ServingOptions(), loaded, served_name, gpus)):
        yield config.LOCAL_VLLM_URL, served_name, loaded


@contextmanager
def running_vllm(args: list[str]) -> Iterator[None]:
    """`vllm serve` with the args, at LOCAL_VLLM_URL until the block ends and frees its GPUs."""
    vllm = subprocess.Popen(["vllm", "serve", *args])
    try:
        wait_until_ready(vllm)
        yield
    finally:
        vllm.terminate()
        vllm.wait()


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
