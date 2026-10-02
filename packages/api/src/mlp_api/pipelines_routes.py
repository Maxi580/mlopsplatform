from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from mlp_api.pipelines.lifecycle import (
    cancel_pipeline,
    get_pipeline,
    list_pipelines,
    submit_pipeline,
)
from mlp_api.pipelines.pipeline_request import validate_pipeline_request
from mlp_core import api_paths
from mlp_core.pipeline_request.schema import PipelineRequest

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
    return {"request": resolve(submission, request).model_dump(mode="json")}


@router.post(api_paths.PIPELINES, status_code=202)
def submit(submission: Submission, request: Request) -> dict:
    resolved = resolve(submission, request)
    state = request.app.state
    try:
        pipeline_id = submit_pipeline(
            state.engine, state.cluster, state.settings, resolved, submission.secrets
        )
    except RuntimeError as error:
        raise HTTPException(502, str(error)) from None
    return {"id": pipeline_id}


@router.get(api_paths.PIPELINES)
def pipelines(request: Request) -> list[dict]:
    return list_pipelines(request.app.state.engine)


# Secrets are never stored, so the resolved request is safe to return.
@router.get(api_paths.PIPELINE)
def pipeline(id: int, request: Request) -> dict:
    try:
        return get_pipeline(request.app.state.engine, id)
    except LookupError as error:
        raise HTTPException(404, str(error)) from None


@router.post(api_paths.CANCEL_PIPELINE)
def cancel(id: int, request: Request) -> dict:
    try:
        cancel_pipeline(request.app.state.engine, request.app.state.cluster, id)
    except LookupError as error:
        raise HTTPException(404, str(error)) from None
    except ValueError as error:
        raise HTTPException(409, str(error)) from None
    return {"id": id, "status": "cancelled"}


# Errors carry paths inside the Pipeline Request, not FastAPI's `body.request` prefix.
def resolve(submission: Submission, request: Request) -> PipelineRequest:
    resolved, errors = validate_pipeline_request(
        submission.request,
        submission.secrets,
        request.app.state.hugging_face,
        request.app.state.engine,
    )
    if errors:
        raise HTTPException(422, errors)
    return resolved
