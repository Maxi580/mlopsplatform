from typing import Annotated

from pydantic import Field

BaseModelReference = Annotated[str, Field(pattern=r"^hf:[\w.-]+/[\w.-]+(@[\w.-]+)?$")]
DatasetReference = Annotated[str, Field(pattern=r"^dataset:[\w.-]+(@\d+)?$")]


def split_base_model_reference(reference: str) -> tuple[str, str]:
    """The Hugging Face repo and revision; the revision is empty when none is named."""
    repo, _, revision = reference.removeprefix("hf:").partition("@")
    return repo, revision


def base_model_reference(repo: str, revision: str) -> str:
    return f"hf:{repo}@{revision}"
