import httpx2 as httpx
import pytest

from mlp_stages import platform_api
from mlp_stages.operations import serve


@pytest.fixture
def api(monkeypatch):
    """Answers the API's serve route with `answer` and records what it received."""
    received = []
    answer = {"status": 201, "json": {"name": "chat", "url": "/endpoints/chat/v1"}}

    def handle(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(answer["status"], json=answer["json"])

    monkeypatch.setenv("API_URL", "http://api.test:8000")
    monkeypatch.setenv("MLP_STEP_TOKEN", "step-token")
    monkeypatch.setattr(
        platform_api, "api_client", lambda: httpx.Client(transport=httpx.MockTransport(handle))
    )
    return received, answer


def test_serve_asks_the_api_with_the_step_token(api, capsys):
    received, _ = api

    serve.serve("7")

    [request] = received
    assert str(request.url) == "http://api.test:8000/pipelines/7/serve"
    assert request.headers["authorization"] == "Bearer step-token"
    assert "/endpoints/chat/v1" in capsys.readouterr().out


def test_a_refusal_fails_the_step_with_the_apis_reason(api):
    _, answer = api
    answer.update(status=409, json={"detail": "Endpoint chat is already running"})

    with pytest.raises(SystemExit, match="Endpoint chat is already running"):
        serve.serve("7")
