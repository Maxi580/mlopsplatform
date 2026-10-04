from pathlib import Path
from typing import Annotated

from pydantic import Field

from mlp_core.config import BENCHMARKS_DIRECTORY

# Starts alphanumeric, so names like `..` never become storage paths.
DATASET_NAME_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_.-]*"
# A Registered Model is named after the Pipeline that produced it.
MODEL_NAME_PATTERN = r"[a-z0-9][a-z0-9.-]*"
BaseModelReference = Annotated[str, Field(pattern=r"^hf:[\w.-]+/[\w.-]+(@[\w.-]+)?$")]
DatasetReference = Annotated[str, Field(pattern=rf"^dataset:{DATASET_NAME_PATTERN}(@\d+)?$")]
ModelReference = Annotated[str, Field(pattern=rf"^model:{MODEL_NAME_PATTERN}(@\d+)?$")]


def split_base_model_reference(reference: str) -> tuple[str, str]:
    """The Hugging Face repo and revision; the revision is empty when none is named."""
    repo, _, revision = reference.removeprefix("hf:").partition("@")
    return repo, revision


def base_model_reference(repo: str, revision: str) -> str:
    return f"hf:{repo}@{revision}"


def split_dataset_reference(reference: str) -> tuple[str, int | None]:
    """The Dataset name and version; the version is None when none is named."""
    name, _, version = reference.removeprefix("dataset:").partition("@")
    return name, int(version) if version else None


def dataset_reference(name: str, version: int) -> str:
    return f"dataset:{name}@{version}"


def split_model_reference(reference: str) -> tuple[str, int | None]:
    """The Registered Model name and version; the version is None when none is named."""
    name, _, version = reference.removeprefix("model:").partition("@")
    return name, int(version) if version else None


def model_reference(name: str, version: int) -> str:
    return f"model:{name}@{version}"


def dataset_key(name: str, version: int) -> str:
    """Where a Dataset Version's JSONL file lives in the platform bucket."""
    return f"datasets/{name}/{version}/data.jsonl"


def split_benchmark_reference(benchmark: str) -> tuple[str, str]:
    """The harness and task of a `harness:task` benchmark."""
    harness, task = benchmark.split(":")
    return harness, task


def benchmark_reference(harness: str, task: str) -> str:
    return f"{harness}:{task}"


def split_endpoint_reference(reference: str) -> str:
    """The Endpoint name an `endpoint:` Reference names."""
    return reference.removeprefix("endpoint:")


def benchmark_directory(model_cache: Path, benchmark: str) -> Path:
    """Where the Model Cache holds a `harness:task` benchmark's datasets."""
    return model_cache / BENCHMARKS_DIRECTORY / Path(*split_benchmark_reference(benchmark))
