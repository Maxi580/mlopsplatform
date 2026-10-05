from mlp_api.endpoints.endpoint_model import find_endpoint_model
from mlp_api.model_cache.janitor import find_cache_entry
from mlp_core import config
from mlp_core.pipeline_request.references import split_base_model_reference
from mlp_core.pipeline_request.schema import PipelineRequest


def fetched_references(state, request: PipelineRequest) -> list[str]:
    """Every Base Model and benchmark the Pipeline's fetch step pulls into the Model Cache."""
    distill, finetune, quantize = request.distill, request.finetune, request.quantize
    speculate, evaluate = request.speculate, request.evaluate
    base_models = []
    if distill and not distill.api_url:
        base_models.append(base_model_of(state, distill.teacher))
    if finetune:
        base_models.append(finetune.base_model)
        base_models += [
            base_model_of(state, phase.teacher) for phase in finetune.phases if phase.teacher
        ]
    if quantize:
        base_models.append(base_model_of(state, quantize.model))
    if speculate:
        base_models.append(base_model_of(state, speculate.model))
    if evaluate:
        base_models.append(base_model_of(state, evaluate.model))
    benchmarks = evaluate.benchmarks if evaluate else []
    return list(dict.fromkeys(reference for reference in base_models + benchmarks if reference))


def base_model_of(state, model: str) -> str | None:
    """The Base Model the model is or is built on; None for an Endpoint or a Stage's output."""
    if model.startswith("hf:"):
        return model
    if model.startswith("model:"):
        found = find_endpoint_model(
            model, state.hugging_face, state.model_registry, state.object_store
        )
        return next(iter(found.base_models), None)
    return None


def preview_downloads(state, references: list[str], token: str | None) -> dict:
    """What fetch pulls into the Model Cache, with the bytes to download and already cached."""
    downloads = [download(state, reference, token) for reference in references]
    return {
        "downloads": downloads,
        "download_bytes": sum(d["bytes"] or 0 for d in downloads if not d["cached"]),
        "cached_bytes": sum(d["bytes"] or 0 for d in downloads if d["cached"]),
    }


# Bytes are None when Hugging Face can't say; a benchmark's come from the catalog.
def download(state, reference: str, token: str | None) -> dict:
    is_base_model = reference.startswith("hf:")
    found = find_cache_entry(state.model_cache, reference)
    cached = found is not None and found.complete
    if cached:
        size = found.size_bytes
    elif is_base_model:
        size = state.hugging_face.model_size(*split_base_model_reference(reference), token)
    else:
        size = config.BENCHMARKS[reference]["size_bytes"]
    kind = "base_model" if is_base_model else "benchmark"
    return {"kind": kind, "ref": reference, "bytes": size, "cached": cached}
