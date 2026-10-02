from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from mlp_api.auth.session import log_in, set_session_cookie
from mlp_api.pipelines.lifecycle import cancel_pipeline, list_pipelines, submit_pipeline
from mlp_api.pipelines.pipeline_request import validate_pipeline_request
from mlp_api.web.pipeline_form import pipeline_form, pipeline_request_from_form
from mlp_core import api_paths, config

router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory=Path(__file__).parent / "web" / "templates")
templates.env.globals.update(api_paths=api_paths, config=config)


@router.get(api_paths.WEB_LOGIN)
def login_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "login.html")


@router.post(api_paths.WEB_LOGIN)
def login(password: Annotated[str, Form()], request: Request) -> HTMLResponse:
    try:
        token = log_in(request, password)
    except HTTPException as error:
        context = {"error": error.detail}
        return templates.TemplateResponse(request, "login.html", context, error.status_code)
    response = RedirectResponse(api_paths.WEB_PIPELINES, status_code=303)
    set_session_cookie(response, token)
    return response


@router.get(api_paths.WEB_PIPELINES)
def pipelines_page(request: Request) -> HTMLResponse:
    fields, form_errors = pipeline_form()
    context = {
        "gpu_count": request.app.state.settings.gpu_count,
        "pipelines": list_pipelines(request.app.state.engine),
        "fields": fields,
        "form_errors": form_errors,
    }
    return templates.TemplateResponse(request, "pipelines.html", context)


@router.get(api_paths.WEB_PIPELINE_LIST)
def pipeline_list(request: Request, notice: str = "") -> HTMLResponse:
    pipelines = list_pipelines(request.app.state.engine)
    context = {"pipelines": pipelines, "notice": notice}
    return templates.TemplateResponse(request, "pipeline_list.html", context)


@router.post(api_paths.WEB_PIPELINE_FORM)
async def submit(request: Request) -> HTMLResponse:
    values = {key: value for key, value in (await request.form()).items() if isinstance(value, str)}
    return await run_in_threadpool(submit_pipeline_form, request, values)


@router.post(api_paths.WEB_CANCEL_PIPELINE)
def cancel(id: int, request: Request) -> HTMLResponse:
    try:
        cancel_pipeline(request.app.state.engine, request.app.state.cluster, id)
        notice = f"Cancelled Pipeline {id}"
    except (LookupError, ValueError) as error:
        notice = str(error)
    return pipeline_list(request, notice)


def submit_pipeline_form(request: Request, values: dict[str, str]) -> HTMLResponse:
    """The form again, with the new Pipeline's ID or every error next to its field."""
    # 1. The Pipeline Request the values describe, checked like `POST /pipelines` checks it.
    state = request.app.state
    data, errors = pipeline_request_from_form(values)
    secrets = {slot: values[slot] for slot in config.SECRET_ENV_VARS if values.get(slot)}
    resolved, request_errors = validate_pipeline_request(
        data, secrets, state.hugging_face, state.engine
    )
    errors += request_errors

    # 2. The Pipeline, unless anything was wrong.
    notice = ""
    if not errors:
        try:
            pipeline_id = submit_pipeline(state.engine, state.cluster, resolved, secrets)
            notice = f"Submitted Pipeline {pipeline_id}"
        except RuntimeError as error:
            errors = [{"loc": [], "msg": str(error)}]

    # 3. The form with the values entered, so only what was wrong needs fixing.
    fields, form_errors = pipeline_form(values, errors)
    context = {"fields": fields, "form_errors": form_errors, "notice": notice}
    response = templates.TemplateResponse(request, "pipeline_form.html", context)
    response.headers["HX-Trigger"] = "pipelines-changed"
    return response
