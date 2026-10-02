from fastapi import APIRouter, Request

from mlp_core import api_paths

router = APIRouter()


# The platform settings, e.g. gpu_count, which the Web UI shows so users know what fits.
@router.get(api_paths.SETTINGS)
def platform_settings(request: Request) -> dict:
    return request.app.state.settings.model_dump()
