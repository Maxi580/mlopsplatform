from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from mlp_api.model_cache.downloads import fetched_references, preview_downloads
from mlp_api.model_cache.janitor import make_room_for_downloads
from mlp_api.pipelines.compiler import Resume
from mlp_api.pipelines.lifecycle import (
    cancel_pipeline,
    get_pipeline,
    list_pipelines,
    submit_pipeline,
)
from mlp_api.pipelines.pipeline_request import validate_pipeline_request
from mlp_api.pipelines.published_schema import published_schema
from mlp_api.pipelines.resume import plan_resume
from mlp_api.pipelines.sweep_output import record_sweep_output
from mlp_core import api_paths, config
from mlp_core.pipeline_request.schema import PipelineRequest

router = APIRouter()


class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: dict[str, Any]
    # Secret values by slot name (e.g. `hf_token`); never stored and never part of the request.
    secrets: dict[str, str] = {}


class ResumeSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Read afresh, as a Pipeline's own are never kept.
    secrets: dict[str, str] = {}


class SweepOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The best Trial's values, as `{settings: …, lora: …}`.
    parameters: dict[str, dict[str, Any]]
    objective: float


@router.get(api_paths.SCHEMA)
def schema() -> dict:
    return published_schema()


# The catalog `evaluate` picks from, with what the Web UI shows beside each benchmark.
@router.get(api_paths.BENCHMARKS)
def benchmarks() -> list[dict]:
    return [{"name": name, **entry} for name, entry in config.BENCHMARKS.items()]


@router.post(api_paths.VALIDATE_PIPELINE)
def validate(submission: Submission, request: Request) -> dict:
    resolved = resolve(submission, request)
    state = request.app.state
    fetched = fetched_references(state, resolved)
    downloads = preview_downloads(state, fetched, submission.secrets.get("hf_token"))
    return {"request": resolved.model_dump(mode="json"), **downloads}


@router.post(api_paths.PIPELINES, status_code=202)
def submit(submission: Submission, request: Request) -> dict:
    return start(submission, request)


# A new Pipeline from the stored resolved request, reusing what the failed or cancelled one made.
@router.post(api_paths.RESUME_PIPELINE, status_code=202)
def resume(id: int, submission: ResumeSubmission, request: Request) -> dict:
    try:
        plan = plan_resume(request.app.state, id)
    except LookupError as error:
        raise HTTPException(404, str(error)) from None
    except ValueError as error:
        raise HTTPException(409, str(error)) from None
    stored = get_pipeline(request.app.state.engine, id)["request"]
    return start(Submission(request=stored, secrets=submission.secrets), request, id, plan)


def start(
    submission: Submission, request: Request, resumed_from=None, plan: Resume | None = None
) -> dict:
    """The started Pipeline's ID and its downloads; 422 if invalid, 502 if Kubeflow refused."""
    # 1. The request resolved, as for a validation.
    resolved = resolve(submission, request)

    # 2. Room in the Model Cache for what fetch downloads.
    state = request.app.state
    fetched = fetched_references(state, resolved)
    downloads = preview_downloads(state, fetched, submission.secrets.get("hf_token"))
    make_room_for_downloads(state, downloads["download_bytes"])

    # 3. The Pipeline and its Kubeflow run.
    try:
        pipeline_id = submit_pipeline(
            state, resolved, submission.secrets, fetched, resumed_from, plan
        )
    except RuntimeError as error:
        raise HTTPException(502, str(error)) from None
    return {"id": pipeline_id, **downloads}


# The Pipeline's `sweep` step; require_login lets only that Pipeline's step token through.
@router.post(api_paths.SWEEP_PIPELINE, status_code=201)
def sweep(id: int, best: SweepOutput, request: Request) -> dict:
    try:
        record_sweep_output(request.app.state.engine, id, best.model_dump())
    except LookupError as error:
        raise HTTPException(404, str(error)) from None
    except ValueError as error:
        raise HTTPException(422, str(error)) from None
    return best.model_dump()


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
        request.app.state.model_registry,
        request.app.state.object_store,
    )
    if errors:
        raise HTTPException(422, errors)
    return resolved
