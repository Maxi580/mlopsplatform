import os

import httpx2 as httpx

from mlp_core import config


def call_step_route(
    path: str, pipeline_id: str, content: bytes | None = None, json: dict | None = None
) -> dict:
    """The API's answer at a step route of the Pipeline, sent with its step token, or exit."""
    token = os.environ[config.SECRET_ENV_VARS["step_token"]]
    with api_client() as client:
        response = client.post(
            os.environ["API_URL"] + path.format(id=pipeline_id),
            content=content,
            json=json,
            headers={"authorization": f"Bearer {token}"},
        )
    if response.is_error:
        raise SystemExit(f"The API refused: {response.json().get('detail')}")
    return response.json()


# Tests swap in a fake API.
def api_client() -> httpx.Client:
    return httpx.Client(timeout=config.STEP_REQUEST_TIMEOUT.total_seconds())
