import json

import httpx2 as httpx
import pytest

from mlp_stages import sandbox
from mlp_stages.operations.check_sandbox import check_sandbox

# What a working Sandbox answers to the step's snippets, in order.
PASSING = [
    {"status": "ok", "stdout": "3\n", "stderr": ""},
    {"status": "ok", "stdout": "True\n", "stderr": ""},
    {"status": "ok", "stdout": "no network\n", "stderr": ""},
    {"status": "memory_limit", "stdout": "", "stderr": "MemoryError"},
    {"status": "timeout", "stdout": "", "stderr": ""},
]


@pytest.fixture
def fake_sandbox(monkeypatch):
    """Answers every batch with `answer` and records what it received."""
    received = []
    answer = [*PASSING]

    def handle(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(200, json=answer)

    monkeypatch.setenv("SANDBOX_URL", "http://sandbox.test:8090")
    monkeypatch.setattr(
        sandbox, "sandbox_client", lambda: httpx.Client(transport=httpx.MockTransport(handle))
    )
    return received, answer


def test_the_check_passes_when_the_sandbox_runs_snippets_within_its_limits(fake_sandbox, capsys):
    received, _ = fake_sandbox

    check_sandbox("1024")

    [request] = received
    assert str(request.url) == "http://sandbox.test:8090/snippets"
    snippets = json.loads(request.content)
    assert f"bytearray({2 * 1024 * 2**20})" in snippets[3]["code"]
    assert "has no network: passed" in capsys.readouterr().out


def test_the_check_fails_naming_each_limit_the_sandbox_did_not_keep(fake_sandbox):
    _, answer = fake_sandbox
    answer[2] = {"status": "ok", "stdout": "network\n", "stderr": ""}

    with pytest.raises(SystemExit, match="has no network"):
        check_sandbox("1024")


def test_a_sandbox_error_fails_the_check(fake_sandbox, monkeypatch):
    monkeypatch.setattr(
        sandbox,
        "sandbox_client",
        lambda: httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(503))),
    )

    with pytest.raises(httpx.HTTPStatusError):
        check_sandbox("1024")
