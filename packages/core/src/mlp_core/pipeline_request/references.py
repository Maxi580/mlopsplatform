from typing import Annotated

from pydantic import Field

# Starts alphanumeric, so names like `..` never become storage paths.
DATASET_NAME_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_.-]*"
BaseModelReference = Annotated[str, Field(pattern=r"^hf:[\w.-]+/[\w.-]+(@[\w.-]+)?$")]
DatasetReference = Annotated[str, Field(pattern=rf"^dataset:{DATASET_NAME_PATTERN}(@\d+)?$")]


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
