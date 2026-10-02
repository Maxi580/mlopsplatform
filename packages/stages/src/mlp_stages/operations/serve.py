import os

import httpx2 as httpx

from mlp_core import api_paths, config


def serve(pipeline_id: str) -> None:
    """Asks the API to start the Endpoint for the Pipeline's output; it outlives the Pipeline."""
    token = os.environ[config.SECRET_ENV_VARS["serve_token"]]
    with api_client() as client:
        response = client.post(
            os.environ["API_URL"] + api_paths.SERVE_PIPELINE.format(id=pipeline_id),
            headers={"authorization": f"Bearer {token}"},
        )
    if response.is_error:
        raise SystemExit(f"The API did not start the Endpoint: {response.json().get('detail')}")
    endpoint = response.json()
    print(f"Endpoint {endpoint['name']} starts at {endpoint['url']}; see the Serving page")


# Tests swap in a fake API.
def api_client() -> httpx.Client:
    return httpx.Client(timeout=config.SERVE_REQUEST_TIMEOUT.total_seconds())
