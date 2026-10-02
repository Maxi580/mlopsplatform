from fastapi import APIRouter, HTTPException, Request

from mlp_api.smoke_tests.cases import COMPLETE_SMOKE_TEST, SmokeTestSelection
from mlp_api.smoke_tests.lifecycle import start_smoke_test
from mlp_core import api_paths

router = APIRouter()


@router.post(api_paths.SMOKE_TEST_COMPLETE, status_code=202)
def complete(request: Request) -> dict:
    return start(request, COMPLETE_SMOKE_TEST)


@router.post(api_paths.SMOKE_TEST_CUSTOM, status_code=202)
def custom(selection: SmokeTestSelection, request: Request) -> dict:
    return start(request, selection)


def start(request: Request, selection: SmokeTestSelection) -> dict:
    try:
        return start_smoke_test(request.app.state, selection)
    except ValueError as error:
        raise HTTPException(409, str(error)) from None
    except RuntimeError as error:
        raise HTTPException(502, str(error)) from None
