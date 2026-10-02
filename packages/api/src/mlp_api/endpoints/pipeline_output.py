from mlp_api.endpoints.lifecycle import start_endpoint
from mlp_api.models.registry import find_pipeline_output
from mlp_api.pipelines.lifecycle import find_pipeline
from mlp_core import config
from mlp_core.endpoint_spec import EndpointSpec
from mlp_core.pipeline_request.references import model_reference
from mlp_core.pipeline_request.schema import PipelineRequest


def serve_pipeline_output(state, pipeline_id: int) -> dict:
    """The Endpoint for the Pipeline's last Model Version, started with its `serve` options."""
    # 1. The Pipeline's own resolved request, while it runs; LookupError if unknown.
    row = find_pipeline(state.engine, pipeline_id)
    if row.status in config.FINISHED_STATUSES:
        raise ValueError(f"Pipeline {pipeline_id} already {row.status}")
    # A Smoke Test keeps each case's request under the case's name; one case serves.
    is_smoke_test = row.cases is not None
    stored = row.request[config.SMOKE_TEST_SERVE_STAGE_CASE] if is_smoke_test else row.request
    request = PipelineRequest.model_validate(stored)
    if request.serve is None:
        raise ValueError(f"Pipeline {pipeline_id} has no `serve` Stage")

    # 2. Its last Model Version, served under the name validation pinned.
    last = find_pipeline_output(state.model_registry, request.name, pipeline_id)
    options = request.serve.model_dump(exclude={"name"})
    spec = EndpointSpec(model=model_reference(last.name, last.version), **options)
    return start_endpoint(state, request.serve.name, spec)
