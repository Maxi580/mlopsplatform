from fastapi import APIRouter, Request

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
