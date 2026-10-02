from mlp_api.model_cache.janitor import find_cached_base_model
from mlp_core.pipeline_request.references import split_base_model_reference


def preview_downloads(state, base_model: str | None, token: str | None) -> dict:
    """What fetch pulls into the Model Cache, with the bytes to download and already cached."""
    downloads = [base_model_download(state, base_model, token)] if base_model else []
    return {
        "downloads": downloads,
        "download_bytes": sum(d["bytes"] or 0 for d in downloads if not d["cached"]),
        "cached_bytes": sum(d["bytes"] or 0 for d in downloads if d["cached"]),
    }


# Bytes are None when Hugging Face can't say.
def base_model_download(state, reference: str, token: str | None) -> dict:
    cached = find_cached_base_model(state.model_cache, reference)
    size = (
        cached.size_bytes
        if cached
        else state.hugging_face.model_size(*split_base_model_reference(reference), token)
    )
    return {"kind": "base_model", "ref": reference, "bytes": size, "cached": cached is not None}
