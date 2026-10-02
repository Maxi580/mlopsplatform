from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from mlp_api.pipeline_request import resolve_pipeline_request
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


# Errors carry paths inside the Pipeline Request, not FastAPI's `body.request` prefix.
@router.post(api_paths.VALIDATE_PIPELINE)
def validate(submission: Submission, request: Request) -> dict:
    resolved, errors = resolve_pipeline_request(
        submission.request, submission.secrets, request.app.state.hugging_face
    )
    if errors:
        raise HTTPException(422, errors)
    return {"request": resolved.model_dump(mode="json")}
