import json

from opentelemetry.trace import Span, Status, StatusCode

from mlp_core import config


class RequestTrace:
    """One chat or completion request's root span, which ends holding its request and answer."""

    def __init__(self, span: Span, body: bytes, source: str | None):
        self.span = span
        self.body = body
        self.source = source if source in config.REQUEST_SOURCES else config.REQUEST_SOURCES[0]
        # The answer as it arrived: one JSON body, or a stream of server-sent events.
        self.answer = bytearray()
        self.trace_id = config.MLFLOW_TRACE_ID.format(
            otel_trace_id=span.get_span_context().trace_id
        )

    def add(self, chunk: bytes) -> None:
        self.answer += chunk

    def finish(self, status: int, error: str | None = None) -> None:
        """Ends the span with the request, the answer, the HTTP status and any error."""
        # 1. The answer as one response, streamed or not; vLLM reports an error in it.
        response = answer_response(bytes(self.answer)) if self.answer else None
        if error is None and (status >= 400 or (response or {}).get("error")):
            error = json.dumps((response or {}).get("error") or response or status)

        # 2. MLflow shows the inputs and outputs and makes `mlflow.traceTag.*` the trace's tags.
        self.span.set_attributes(
            {
                config.MLFLOW_SPAN_TYPE: json.dumps("CHAT_MODEL"),
                config.MLFLOW_SPAN_INPUTS: json.dumps(json_or_text(self.body)),
                config.MLFLOW_SPAN_OUTPUTS: json.dumps(response),
                "http.status_code": status,
                config.MLFLOW_TRACE_TAG.format(key="source"): self.source,
                config.MLFLOW_TRACE_TAG.format(key="status"): "error" if error else "ok",
            }
        )
        if error:
            self.span.set_attribute("error", error)
            self.span.set_status(Status(StatusCode.ERROR, error))
        self.span.end()


def answer_response(answer: bytes) -> dict | str:
    """A JSON answer as it is; a streamed one assembled as vLLM answers without streaming."""
    if not answer.lstrip().startswith(b"data:"):
        return json_or_text(answer)
    chunks = [
        json.loads(data)
        for line in answer.decode(errors="replace").splitlines()
        if line.startswith("data:") and (data := line.removeprefix("data:").strip()) != "[DONE]"
    ]
    return assembled_response(chunks)


def json_or_text(body: bytes) -> dict | str:
    try:
        return json.loads(body)
    except ValueError:
        return body.decode(errors="replace")


def assembled_response(chunks: list[dict]) -> dict:
    """The streamed chunks as one chat or text completion, with their usage and any error."""
    choices: dict[int, dict] = {}
    for chunk in chunks:
        for delta_choice in chunk.get("choices", []):
            choice = choices.setdefault(delta_choice["index"], {"index": delta_choice["index"]})
            if "delta" in delta_choice:
                add_delta(choice, delta_choice["delta"])
            else:
                choice["text"] = choice.get("text", "") + (delta_choice.get("text") or "")
            if delta_choice.get("finish_reason"):
                choice["finish_reason"] = delta_choice["finish_reason"]
    for choice in choices.values():
        calls = choice.get("message", {}).get("tool_calls")
        if calls:
            choice["message"]["tool_calls"] = [calls[index] for index in sorted(calls)]
    first = chunks[0] if chunks else {}
    response = {
        "id": first.get("id"),
        "object": (first.get("object") or "").removesuffix(".chunk"),
        "model": first.get("model"),
        "choices": [choices[index] for index in sorted(choices)],
        "usage": next((c["usage"] for c in reversed(chunks) if c.get("usage")), None),
    }
    error = next((c["error"] for c in chunks if c.get("error")), None)
    return {**response, "error": error} if error else response


def add_delta(choice: dict, delta: dict) -> None:
    """Adds a chat chunk's content, reasoning and tool call parts to the choice's message."""
    message = choice.setdefault("message", {"role": "assistant", "content": ""})
    message["content"] += delta.get("content") or ""
    # Older vLLMs name it `reasoning_content`, newer ones `reasoning`.
    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
    if reasoning:
        message["reasoning_content"] = message.get("reasoning_content", "") + reasoning
    for part in delta.get("tool_calls") or []:
        empty = {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
        call = message.setdefault("tool_calls", {}).setdefault(part["index"], empty)
        call["id"] = part.get("id") or call["id"]
        function = part.get("function") or {}
        call["function"]["name"] += function.get("name") or ""
        call["function"]["arguments"] += function.get("arguments") or ""
