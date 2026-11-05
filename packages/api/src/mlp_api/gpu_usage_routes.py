from fastapi import APIRouter, Request

from mlp_api.gpu_usage.machines import fullest_machine_reserved_percent
from mlp_core import api_paths

router = APIRouter()


# What the line under each GPU Share field says.
@router.get(api_paths.GPU_RESERVED)
def gpu_reserved(request: Request) -> dict:
    state = request.app.state
    return {
        "reserved_percent": fullest_machine_reserved_percent(state.cluster, state.settings),
        "recommended_max": state.settings.gpu_max_utilization,
    }
