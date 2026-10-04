from mlp_core import api_paths
from mlp_stages.platform_api import call_step_route


def serve(pipeline_id: str) -> None:
    """Asks the API to start the Endpoint for the Pipeline's output; it outlives the Pipeline."""
    endpoint = call_step_route(api_paths.SERVE_PIPELINE, pipeline_id)
    print(f"Endpoint {endpoint['name']} starts at {endpoint['url']}; see the Serving page")
