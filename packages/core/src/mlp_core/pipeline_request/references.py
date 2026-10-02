from typing import Annotated

from pydantic import Field

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
