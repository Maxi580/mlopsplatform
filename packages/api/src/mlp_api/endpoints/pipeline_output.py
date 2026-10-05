from mlp_api.endpoints.lifecycle import start_endpoint
from mlp_api.models.registry import find_pipeline_output
from mlp_api.pipelines.compiler import Resume
from mlp_api.pipelines.lifecycle import find_pipeline, find_running_request
from mlp_core import config
from mlp_core.endpoint_spec import EndpointSpec
from mlp_core.pipeline_request.references import model_reference


def serve_pipeline_output(state, pipeline_id: int) -> dict:
    """The Endpoint for the Pipeline's last Model Version, started with its `serve` options."""
    # 1. The Pipeline's own resolved request, while it runs; LookupError if unknown.
    case = config.SMOKE_TEST_SERVE_STAGE_CASE
    request = find_running_request(state.engine, pipeline_id, case)
    if request.serve is None:
        raise ValueError(f"Pipeline {pipeline_id} has no `serve` Stage")

    # 2. Its last Model Version, or the last one it reused if a resume found every Phase finished.
    try:
        last = find_pipeline_output(state.model_registry, request.name, pipeline_id)
        model = model_reference(last.name, last.version)
    except ValueError:
        reused = Resume(**(find_pipeline(state.engine, pipeline_id).resume or {}))
        if not reused.model_versions:
            raise
        model = reused.model_versions[-1]

    # 3. Served under the name validation pinned.
    options = request.serve.model_dump(exclude={"name"})
    spec = EndpointSpec(model=model, **options)
    return start_endpoint(state, request.serve.name, spec)
