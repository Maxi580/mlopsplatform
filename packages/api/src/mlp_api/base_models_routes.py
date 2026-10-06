from fastapi import APIRouter, Header, Request

from mlp_api.pipelines.hugging_face import search_base_models
from mlp_core import api_paths

router = APIRouter()


# The Web UI's Base Model picker; the browser never asks huggingface.co itself.
@router.get(api_paths.BASE_MODELS)
def base_models(
    request: Request, search: str = "", x_hf_token: str | None = Header(None)
) -> list[dict]:
    return search_base_models(request.app.state.hugging_face, search, x_hf_token)
