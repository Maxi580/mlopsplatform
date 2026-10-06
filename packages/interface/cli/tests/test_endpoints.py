from typer.testing import CliRunner

from mlp_cli.main import app
from mlp_core import api_paths

from .conftest import TOKEN

ENDPOINT = {
    "name": "chat",
    "owner": "shared",
    "model": "model:qwen-sft@1",
    "status": "running",
    "url": "/endpoints/chat/v1",
    "created_at": "2026-10-01T08:30:00+00:00",
}


def mlp_endpoints(*args):
    return CliRunner().invoke(app, ["endpoints", *args])


def test_endpoints_lists_each_with_its_model_status_and_full_url(logged_in, fake_api):
    fake_api.answers[api_paths.ENDPOINTS] = (200, [ENDPOINT])

    result = mlp_endpoints()

    assert result.exit_code == 0, result.output
    row = result.output.splitlines()[1]
    for text in ("chat", "model:qwen-sft@1", "running", f"{fake_api.url}/endpoints/chat/v1"):
        assert text in row


def test_start_sends_the_model_and_serving_options_and_prints_the_url(logged_in, fake_api):
    fake_api.answers[api_paths.ENDPOINTS] = (201, {**ENDPOINT, "status": "pending"})

    result = mlp_endpoints(
        "start",
        "model:qwen-sft@1",
        "--name",
        "chat",
        "-o",
        "max_model_len=4096",
        "-o",
        "prefix_caching=false",
        "-o",
        "kv_cache_dtype=fp8",
    )

    assert result.exit_code == 0, result.output
    assert fake_api.received == [
        (
            api_paths.ENDPOINTS,
            {
                "name": "chat",
                "model": "model:qwen-sft@1",
                "max_model_len": 4096,
                "prefix_caching": False,
                "kv_cache_dtype": "fp8",
            },
        )
    ]
    assert f"{fake_api.url}/endpoints/chat/v1" in result.output
    assert "pending" in result.output


def test_a_rejected_start_prints_why(logged_in, fake_api):
    detail = [{"loc": ["body", "tensor_parallel_size"], "msg": "Extra inputs are not permitted"}]
    fake_api.answers[api_paths.ENDPOINTS] = (422, {"detail": detail})

    result = mlp_endpoints(
        "start", "hf:Qwen/Qwen3-0.6B", "--name", "chat", "-o", "tensor_parallel_size=2"
    )

    assert result.exit_code == 1
    assert "tensor_parallel_size" in result.output


def test_an_option_without_a_value_is_refused_before_calling_the_api(logged_in, fake_api):
    result = mlp_endpoints("start", "hf:Qwen/Qwen3-0.6B", "--name", "chat", "-o", "dtype")

    assert result.exit_code != 0
    assert fake_api.received == []


def test_stop_stops_the_endpoint(logged_in, fake_api):
    path = api_paths.STOP_ENDPOINT.format(name="chat")
    fake_api.answers[path] = (200, {"name": "chat", "status": "stopped"})

    result = mlp_endpoints("stop", "chat")

    assert result.exit_code == 0, result.output
    assert fake_api.received == [(path, b"")]
    assert "Stopped chat" in result.output


def test_delete_deletes_the_stopped_endpoint(logged_in, fake_api):
    path = api_paths.ENDPOINT.format(name="chat")
    fake_api.answers[path] = (204, b"")

    result = mlp_endpoints("delete", "chat")

    assert result.exit_code == 0, result.output
    assert fake_api.received == [(path, f"Bearer {TOKEN}")]
    assert "Deleted chat" in result.output


def test_deleting_a_running_endpoint_says_why_it_was_refused(logged_in, fake_api):
    detail = "Endpoint chat is running; stop it first"
    fake_api.answers[api_paths.ENDPOINT.format(name="chat")] = (409, {"detail": detail})

    result = mlp_endpoints("delete", "chat")

    assert result.exit_code == 1
    assert detail in result.output


def test_endpoints_send_the_login_token(logged_in, fake_api):
    fake_api.answers[api_paths.ENDPOINTS] = (200, [])

    result = mlp_endpoints()

    assert result.exit_code == 0, result.output
    assert fake_api.received == [(api_paths.ENDPOINTS, f"Bearer {TOKEN}")]
