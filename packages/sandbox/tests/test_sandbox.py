import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mlp_core import config
from mlp_sandbox.server import app

# The Sandbox limits snippets with Linux process limits; on the cluster it also runs under gVisor.
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the Sandbox runs on Linux")


@pytest.fixture
def sandbox(settings_configmap_env, monkeypatch):
    monkeypatch.setenv("sandbox_timeout_seconds", "2")
    monkeypatch.setenv("sandbox_memory_mb", "256")
    with TestClient(app) as client:
        yield client


def run(sandbox, *snippets) -> list[dict]:
    response = sandbox.post(config.SANDBOX_PATH, json=list(snippets))
    assert response.status_code == 200, response.text
    return response.json()


def test_a_batch_returns_each_snippets_result_in_order(sandbox):
    ran, raised = run(
        sandbox,
        {"code": "print(input().upper())", "input": "hi"},
        {"code": "raise ValueError('x')"},
    )

    assert ran == {"status": "ok", "stdout": "HI\n", "stderr": ""}
    assert raised["status"] == "error"
    assert "ValueError: x" in raised["stderr"]


def test_a_snippet_over_the_timeout_is_killed_without_affecting_the_others(sandbox):
    slow, fast = run(
        sandbox, {"code": "import time; print('started'); time.sleep(60)"}, {"code": "print(1)"}
    )

    assert slow == {"status": "timeout", "stdout": "started\n", "stderr": ""}
    assert fast["status"] == "ok"


def test_the_timeout_also_kills_the_processes_a_snippet_started(sandbox):
    code = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    [result] = run(sandbox, {"code": code})

    assert result["status"] == "timeout"
    assert not is_running(int(result["stdout"]))


def test_a_snippet_over_the_memory_limit_is_stopped_without_affecting_the_others(sandbox):
    large, small = run(
        sandbox, {"code": "data = bytearray(512 * 2**20)"}, {"code": "data = bytearray(2**20)"}
    )

    assert large["status"] == "memory_limit"
    assert small["status"] == "ok"


def test_a_librarys_own_memory_error_also_counts_as_the_memory_limit(sandbox):
    code = "class ArrayMemoryError(MemoryError): pass\nraise ArrayMemoryError('Unable to allocate')"

    [result] = run(sandbox, {"code": code})

    assert result["status"] == "memory_limit"


def test_output_beyond_the_limit_stops_the_snippet(sandbox):
    [result] = run(sandbox, {"code": f"print('x' * {2 * config.SANDBOX_OUTPUT_LIMIT_BYTES})"})

    assert result["status"] == "error"
    assert len(result["stdout"]) <= config.SANDBOX_OUTPUT_LIMIT_BYTES


def test_snippets_see_none_of_the_sandboxs_environment(sandbox, monkeypatch):
    monkeypatch.setenv("SOME_SECRET", "value")

    [result] = run(sandbox, {"code": "import os; print(sorted(os.environ))"})

    assert "SOME_SECRET" not in result["stdout"]


def test_each_snippet_starts_in_a_fresh_empty_directory(sandbox):
    run(sandbox, {"code": "open('left-behind', 'w').write('x')"})

    [result] = run(sandbox, {"code": "import os; print(os.listdir('.'))"})

    assert result["stdout"] == "[]\n"


def test_unknown_snippet_fields_are_rejected(sandbox):
    response = sandbox.post(config.SANDBOX_PATH, json=[{"code": "print(1)", "network": True}])

    assert response.status_code == 422


def is_running(pid: int) -> bool:
    """Whether the process exists and is not a zombie waiting for its parent."""
    status = Path(f"/proc/{pid}/status")
    return status.exists() and "State:\tZ" not in status.read_text()
