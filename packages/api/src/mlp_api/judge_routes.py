from fastapi import APIRouter, HTTPException, Request

from mlp_api.endpoints.lifecycle import EndpointConflict
from mlp_api.judge.lifecycle import judge_summary, save_judge_settings
from mlp_api.judge.settings import JudgeSettings
from mlp_core import api_paths

router = APIRouter()


@router.get(api_paths.JUDGE)
def judge(request: Request) -> dict:
    return judge_summary(request.app.state)


@router.put(api_paths.JUDGE)
def save(settings: JudgeSettings, request: Request) -> dict:
    try:
        save_judge_settings(request.app.state, settings)
    except EndpointConflict as error:
        raise HTTPException(409, str(error)) from None
    except ValueError as error:
        raise HTTPException(422, f"Can't serve the Judge's model: {error}") from None
    except RuntimeError as error:
        raise HTTPException(502, str(error)) from None
    return judge_summary(request.app.state)
