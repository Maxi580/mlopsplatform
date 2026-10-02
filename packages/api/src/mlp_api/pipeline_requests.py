from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from mlp_api.config import DEFAULT_HF_REVISION, HF_TOKEN_PATTERN, MIN_SECRET_LENGTH
from mlp_api.hugging_face import HuggingFace
from mlp_core import api_paths
from mlp_core.pipeline_request.schema import PipelineRequest
from mlp_core.pipeline_request.validate import validate_pipeline_request
from mlp_core.settings import Settings

router = APIRouter()


class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: dict[str, Any]
    # Secret values by slot name (e.g. `hf_token`); never stored and never part of the request.
    secrets: dict[str, str] = {}


@router.get(api_paths.SCHEMA)
def schema() -> dict:
    return PipelineRequest.model_json_schema()


@router.post(api_paths.VALIDATE_PIPELINE)
def validate(submission: Submission, request: Request) -> dict:
    resolved = resolve_pipeline_request(
        submission, request.app.state.settings, request.app.state.hugging_face
    )
    return {"request": resolved.model_dump(mode="json")}


def resolve_pipeline_request(
    submission: Submission, settings: Settings, hugging_face: HuggingFace
) -> PipelineRequest:
    """The request with every reference pinned, or a 422 listing every error with its path."""
    request, errors = validate_pipeline_request(submission.request)
    errors += [
        *remote_code_errors(submission.request),
        *secret_value_errors(submission.request, submission.secrets),
        *gpu_errors(settings),
    ]
    if request:
        base_model, base_model_errors = pin_base_model(request, submission.secrets, hugging_face)
        errors += base_model_errors
    if errors:
        raise HTTPException(
            422, [{**error, "loc": ["body", "request", *error["loc"]]} for error in errors]
        )
    finetune = request.finetune.model_copy(update={"base_model": base_model})
    return request.model_copy(update={"finetune": finetune})


def error(loc: list, msg: str) -> dict:
    return {"loc": loc, "msg": msg}


def nodes(node, loc: list) -> Iterator[tuple[list, object]]:
    yield loc, node
    if isinstance(node, dict):
        for key, value in node.items():
            yield from nodes(value, [*loc, key])
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from nodes(value, [*loc, index])


def remote_code_errors(request: dict) -> Iterator[dict]:
    for loc, _ in nodes(request, []):
        if loc and loc[-1] == "trust_remote_code":
            yield error(loc, "`trust_remote_code` is never allowed: repository code doesn't run")


def secret_value_errors(request: dict, secrets: dict[str, str]) -> Iterator[dict]:
    values = [value for value in secrets.values() if len(value) >= MIN_SECRET_LENGTH]
    for loc, node in nodes(request, []):
        if isinstance(node, str) and (
            HF_TOKEN_PATTERN.search(node) or any(value in node for value in values)
        ):
            yield error(loc, "contains a Secret value; name the Secret instead")


def gpu_errors(settings: Settings) -> Iterator[dict]:
    if settings.gpus_per_stage > settings.gpu_count:
        yield error(
            ["finetune"],
            f"finetune needs {settings.gpus_per_stage} GPUs, the platform has {settings.gpu_count}",
        )


def pin_base_model(
    request: PipelineRequest, secrets: dict[str, str], hugging_face: HuggingFace
) -> tuple[str, list[dict]]:
    reference = request.finetune.base_model
    repo, _, revision = reference.removeprefix("hf:").partition("@")
    model = hugging_face.find_model(repo, revision or DEFAULT_HF_REVISION, secrets.get("hf_token"))
    loc = ["finetune", "base_model"]
    if model is None:
        reason = "is missing, gated for this token, or has no such revision"
        return reference, [error(loc, f"{repo} on Hugging Face {reason}")]
    if model.needs_remote_code:
        return reference, [error(loc, f"{repo} needs remote code, which never runs here")]
    return f"hf:{repo}@{model.commit}", []
