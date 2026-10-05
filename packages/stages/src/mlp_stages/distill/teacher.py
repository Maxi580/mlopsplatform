import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx2 as httpx

from mlp_core import config
from mlp_core.pipeline_request.schema import PipelineRequest
from mlp_stages.served_model import served_model


def teacher_replies(
    request: PipelineRequest, prompts: list[list[dict]], teacher_url: str, gpus: int, scratch: Path
) -> list[dict]:
    """The Teacher's reply to each prompt, an OpenAI chat choice, or `{"error": …}` if none came."""
    # 1. What every prompt is sent with: the Teacher's name, the tools and the sampling settings.
    distill = request.distill
    body = {"max_tokens": distill.max_tokens, "temperature": distill.temperature}
    body = {key: value for key, value in body.items() if value is not None}
    if distill.tools:
        body["tools"] = [tool.model_dump(exclude_none=True) for tool in distill.tools]
        body["parallel_tool_calls"] = distill.parallel_tool_calls

    # 2. An API Teacher, with its key.
    if distill.api_url:
        api_key = os.environ[config.SECRET_ENV_VARS["teacher_api_key"]]
        headers = {"authorization": f"Bearer {api_key}"}
        body["model"] = distill.teacher
        return ask_teacher(distill.api_url.rstrip("/"), headers, body, prompts, api_key)

    # 3. Otherwise a running Endpoint, or a vLLM started here with the Teacher's tool parser.
    served = served_model(
        distill.teacher, distill.serving, request.name, teacher_url, gpus, scratch
    )
    with served as (url, served_name, _):
        return ask_teacher(f"{url}/v1", {}, {**body, "model": served_name}, prompts)


def ask_teacher(
    url: str, headers: dict, body: dict, prompts: list[list[dict]], api_key: str = ""
) -> list[dict]:
    """Each prompt's reply, asking several at once."""
    with teacher_client(headers) as client:

        def ask(messages: list[dict]) -> dict:
            return ask_once(client, url, {**body, "messages": messages}, api_key)

        with ThreadPoolExecutor(config.DISTILL_CONCURRENT_REQUESTS) as pool:
            return list(pool.map(ask, prompts))


# Tests swap in a fake Teacher.
def teacher_client(headers: dict) -> httpx.Client:
    return httpx.Client(headers=headers, timeout=config.DISTILL_REPLY_TIMEOUT.total_seconds())


def ask_once(client: httpx.Client, url: str, body: dict, api_key: str) -> dict:
    try:
        response = client.post(f"{url}/chat/completions", json=body)
        if response.is_error:
            # A refusal may quote the key, and the Run's dropped replies aren't redacted.
            reason = f"{response.status_code} {response.text}"
            return {"error": reason.replace(api_key, "***") if api_key else reason}
        return response.json()["choices"][0]
    except (httpx.HTTPError, ValueError, KeyError, IndexError) as error:
        return {"error": f"{type(error).__name__}: {error}"}
