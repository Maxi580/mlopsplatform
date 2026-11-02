import json

import httpx2
import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mlp_proxy import server

VLLM = "http://endpoint-chat.mlp.svc:8000"
ROUTED = {"X-Mlp-Upstream": VLLM, "X-Mlp-Experiment": "7"}
CHAT = {"model": "chat", "messages": [{"role": "user", "content": "Say hi."}], "temperature": 0.2}
ANSWER = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "model": "chat",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hi!", "reasoning_content": "Greet."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}


def sse(*chunks: dict) -> bytes:
    return b"".join(f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks) + (
        b"data: [DONE]\n\n"
    )


def chunk(delta: dict, finish_reason=None, **extra) -> dict:
    choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
    return {"id": "chatcmpl-1", "object": "chat.completion.chunk", "model": "chat", **extra} | {
        "choices": [choice]
    }


@pytest.fixture
def vllm(monkeypatch):
    """The upstream vLLM: answers from `answers` by path, and records each request it got."""
    calls, answers = [], {}

    def answer(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        status, body, content_type = answers[request.url.path]
        headers = {"content-type": content_type}
        return httpx2.Response(status, stream=httpx2.ByteStream(body), headers=headers)

    transport = httpx2.MockTransport(answer)
    monkeypatch.setattr(server, "client", httpx2.AsyncClient(transport=transport))
    return calls, answers


@pytest.fixture
def spans(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracers = {}
    monkeypatch.setattr(
        server, "tracer", lambda found: tracers.setdefault(found, provider.get_tracer("test"))
    )
    return exporter, tracers


@pytest.fixture
def proxy():
    return TestClient(server.app)


def test_a_chat_request_goes_to_vllm_and_its_trace_holds_the_request_and_answer(proxy, vllm, spans):
    calls, answers = vllm
    answers["/v1/chat/completions"] = (200, json.dumps(ANSWER).encode(), "application/json")

    response = proxy.post(
        "/v1/chat/completions", json=CHAT, headers={**ROUTED, "Authorization": "Bearer k"}
    )

    assert response.status_code == 200
    assert response.json() == ANSWER
    [span] = spans[0].get_finished_spans()
    assert response.headers["x-mlp-trace-id"] == f"tr-{span.context.trace_id:032x}"
    assert list(spans[1]) == ["7"]
    assert json.loads(span.attributes["mlflow.spanInputs"]) == CHAT
    assert json.loads(span.attributes["mlflow.spanOutputs"]) == ANSWER
    assert span.attributes["mlflow.traceTag.source"] == "api"
    assert span.attributes["mlflow.traceTag.status"] == "ok"
    assert span.attributes["http.status_code"] == 200
    # vLLM's own span joins the trace; the routing headers and the Endpoint Key stay behind.
    [sent] = calls
    trace_id, parent_id = sent.headers["traceparent"].split("-")[1:3]
    assert (int(trace_id, 16), int(parent_id, 16)) == (span.context.trace_id, span.context.span_id)
    assert not {"x-mlp-upstream", "x-mlp-experiment", "authorization"} & set(sent.headers)
    assert str(sent.url) == f"{VLLM}/v1/chat/completions"


def test_a_streamed_answer_streams_through_and_its_trace_holds_it_assembled(proxy, vllm, spans):
    _, answers = vllm
    tool_call = {"index": 0, "id": "call-1", "function": {"name": "weather", "arguments": '{"c'}}
    stream = sse(
        chunk({"role": "assistant", "content": ""}),
        chunk({"reasoning_content": "Think"}),
        chunk({"reasoning": "ing."}),
        chunk({"content": "Hi"}),
        chunk({"content": "!", "tool_calls": [tool_call]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'ity": "Rome"}'}}]}, "stop"),
        {**chunk({}), "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 4}},
    )
    answers["/v1/chat/completions"] = (200, stream, "text/event-stream")

    response = proxy.post(
        "/v1/chat/completions",
        json={**CHAT, "stream": True},
        headers={**ROUTED, "X-Mlp-Source": "chat"},
    )

    assert response.content == stream
    [span] = spans[0].get_finished_spans()
    assert json.loads(span.attributes["mlflow.spanOutputs"]) == {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "chat",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hi!",
                    "reasoning_content": "Thinking.",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "weather", "arguments": '{"city": "Rome"}'},
                        }
                    ],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 4},
    }
    assert span.attributes["mlflow.traceTag.source"] == "chat"


def test_a_streamed_completion_is_assembled_as_text(proxy, vllm, spans):
    _, answers = vllm
    parts = [{"index": 0, "text": "Hel"}, {"index": 0, "text": "lo", "finish_reason": "length"}]
    stream = sse(*({"object": "text_completion", "choices": [part]} for part in parts))
    answers["/v1/completions"] = (200, stream, "text/event-stream")

    proxy.post("/v1/completions", json={"prompt": "Say", "stream": True}, headers=ROUTED)

    [span] = spans[0].get_finished_spans()
    [choice] = json.loads(span.attributes["mlflow.spanOutputs"])["choices"]
    assert choice == {"index": 0, "text": "Hello", "finish_reason": "length"}


def test_an_answer_vllm_refused_is_traced_as_an_error(proxy, vllm, spans):
    _, answers = vllm
    refusal = {"error": {"message": "max_tokens is too large", "code": 400}}
    answers["/v1/chat/completions"] = (400, json.dumps(refusal).encode(), "application/json")

    response = proxy.post("/v1/chat/completions", json=CHAT, headers=ROUTED)

    assert response.status_code == 400
    [span] = spans[0].get_finished_spans()
    assert span.attributes["mlflow.traceTag.status"] == "error"
    assert "max_tokens is too large" in span.attributes["error"]
    assert not span.status.is_ok


def test_other_paths_pass_through_untraced(proxy, vllm, spans):
    calls, answers = vllm
    answers["/v1/models"] = (200, b'{"data": []}', "application/json")

    response = proxy.get("/v1/models?limit=1", headers=ROUTED)

    assert response.json() == {"data": []}
    assert calls[0].url.params["limit"] == "1"
    assert spans[0].get_finished_spans() == ()
    assert "x-mlp-trace-id" not in response.headers


@pytest.mark.parametrize("missing", ["X-Mlp-Upstream", "X-Mlp-Experiment"])
def test_a_request_without_its_routing_headers_is_refused(proxy, vllm, spans, missing):
    headers = {key: value for key, value in ROUTED.items() if key != missing}

    response = proxy.post("/v1/chat/completions", json=CHAT, headers=headers)

    assert response.status_code == 502
    assert vllm[0] == []


def test_a_chat_request_while_mlflow_is_unreachable_is_served_untraced(proxy, vllm, spans):
    _, answers = vllm
    answers["/v1/chat/completions"] = (200, json.dumps(ANSWER).encode(), "application/json")
    headers = {**ROUTED, "X-Mlp-Experiment": "none"}

    response = proxy.post("/v1/chat/completions", json=CHAT, headers=headers)

    assert response.json() == ANSWER
    assert spans[0].get_finished_spans() == ()
    assert "x-mlp-trace-id" not in response.headers
