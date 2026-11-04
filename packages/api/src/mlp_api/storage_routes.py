from fastapi import APIRouter, Request

from mlp_api.gpu_usage.machines import gpu_machines
from mlp_core import api_paths
from mlp_core.settings import size_in_bytes

router = APIRouter()


# Object store usage per bucket, so users know when to clean up.
@router.get(api_paths.STORAGE)
def storage(request: Request) -> dict:
    sizes = request.app.state.object_store.bucket_sizes()
    return {
        "buckets": [{"name": name, "size_bytes": sizes[name]} for name in sorted(sizes)],
        "capacity_bytes": size_in_bytes(request.app.state.settings.object_store_size),
    }


# How full each GPU is right now, so users see why a job waits and how large a share to pick.
@router.get(api_paths.STORAGE_GPUS)
def storage_gpus(request: Request) -> dict:
    return {"machines": gpu_machines(request.app.state.cluster, request.app.state.settings)}
