import os

import httpx2 as httpx

from mlp_core import config


def run_in_sandbox(snippets: list[dict]) -> list[dict]:
    """Each snippet's status, stdout and stderr from the Sandbox, in order."""
    with sandbox_client() as client:
        response = client.post(os.environ["SANDBOX_URL"] + config.SANDBOX_PATH, json=snippets)
    response.raise_for_status()
    return response.json()


# A batch takes as long as its snippets; tests swap in a fake Sandbox.
def sandbox_client() -> httpx.Client:
    return httpx.Client(timeout=None)
