import json

from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match

from mlp_core.pipeline_request.schema import Distill


def distilled_row(prompt: list[dict], reply: dict, distill: Distill) -> dict:
    """The prompt and the Teacher's reply as a prompt-completion row; ValueError if malformed."""
    # 1. A whole reply: a text, or tool calls.
    if "error" in reply:
        raise ValueError(f"the Teacher gave no reply: {reply['error']}")
    if reply.get("finish_reason") == "length":
        raise ValueError("the reply was cut off at max_tokens")
    message = reply.get("message") or {}
    content, calls = message.get("content") or "", message.get("tool_calls") or []
    if not calls and not content.strip():
        raise ValueError("the reply is empty")
    if len(calls) > 1 and not distill.parallel_tool_calls:
        raise ValueError("the reply makes several tool calls, and parallel_tool_calls is off")

    # 2. Each call to an offered tool, with arguments that match its schema, stored as a dict.
    tools = {tool.function.name: tool.function.parameters for tool in distill.tools or []}
    completion = {"role": "assistant", "content": content}
    if calls:
        completion["tool_calls"] = [checked_tool_call(call, tools) for call in calls]
    row = {"prompt": prompt, "completion": [completion]}
    if distill.tools:
        row["tools"] = [tool.model_dump(exclude_none=True) for tool in distill.tools]
    return row


def checked_tool_call(call: dict, tools: dict[str, dict]) -> dict:
    """The call with its arguments as a dict; ValueError unless it fits an offered tool."""
    function = call.get("function") or {}
    name, arguments = function.get("name"), function.get("arguments")
    if name not in tools:
        raise ValueError(f"the reply calls `{name}`, which isn't offered")
    # OpenAI's wire format sends the arguments as a JSON string.
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            raise ValueError(f"the arguments of `{name}` are no JSON") from None
    if not isinstance(arguments, dict):
        raise ValueError(f"the arguments of `{name}` are no JSON object")
    mismatch = best_match(Draft202012Validator(tools[name]).iter_errors(arguments))
    if mismatch is not None:
        raise ValueError(f"the arguments of `{name}` don't match its schema: {mismatch.message}")
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


def prompt_messages(prompt: str | list[dict]) -> list[dict]:
    """The prompt as messages, so that the Teacher's reply message can follow it."""
    return [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
