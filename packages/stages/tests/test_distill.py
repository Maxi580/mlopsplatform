import json
import sys
from types import SimpleNamespace

import httpx2 as httpx
import pytest

mlflow = pytest.importorskip("mlflow", reason="needs the evaluate extra: uv sync --extra evaluate")

from mlp_stages import platform_api, served_model  # noqa: E402
from mlp_stages.distill import main as distill_main  # noqa: E402
from mlp_stages.distill import teacher  # noqa: E402
from mlp_stages.main import main  # noqa: E402

from .test_evaluate import BASE_MODEL, REPO, FakeMlflow, FakeVllm, option  # noqa: E402

TEACHER_API_KEY = "sk-teacher-key-0123456789"
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def text(content: str) -> dict:
    return {"finish_reason": "stop", "message": {"role": "assistant", "content": content}}


def calls(*calls: tuple[str, object]) -> dict:
    """A reply calling each named tool with its arguments, as a JSON string unless given raw."""
    tool_calls = [
        {
            "id": f"call-{index}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
            },
        }
        for index, (name, arguments) in enumerate(calls)
    ]
    message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
    return {"finish_reason": "tool_calls", "message": message}


class FakeTeacher:
    """An OpenAI-compatible server answering each prompt's last message with its set reply."""

    def __init__(self):
        self.replies = {}
        self.requests = []
        self.status = 200

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": f"bad key {TEACHER_API_KEY}"})
        body = json.loads(request.content)
        reply = self.replies[body["messages"][-1]["content"]]
        return httpx.Response(200, json={"choices": [reply]})


class FakeApi:
    """The API's distill route, recording the rows it was sent."""

    def __init__(self):
        self.requests = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(201, json={"dataset": "dataset:qwen-distill@1"})

    @property
    def rows(self) -> list[dict]:
        [request] = self.requests
        return [json.loads(line) for line in request.content.decode().splitlines()]


class DistillMlflow(FakeMlflow):
    """Also keeps the artifacts logged into the step's Run."""

    def __init__(self):
        super().__init__()
        self.artifacts = {}

    def log_dict(self, run_id, dictionary, artifact_file):
        self.artifacts[artifact_file] = dictionary


@pytest.fixture
def step(monkeypatch, tmp_path):
    """Runs `mlp-stage distill` on the set prompts, with fakes for vLLM, the Teacher and the API."""
    fakes = SimpleNamespace(
        vllm=FakeVllm(), teacher=FakeTeacher(), api=FakeApi(), mlflow=DistillMlflow(), prompts=[]
    )
    output = tmp_path / "outputs" / "dataset"
    fakes.output = output
    monkeypatch.setattr(served_model.subprocess, "Popen", fakes.vllm)
    monkeypatch.setattr(served_model.httpx, "get", lambda url: SimpleNamespace(is_success=True))
    monkeypatch.setattr(mlflow, "MlflowClient", fakes.mlflow)
    transport = httpx.MockTransport(fakes.teacher.handle)
    monkeypatch.setattr(
        teacher,
        "teacher_client",
        lambda headers: httpx.Client(transport=transport, headers=headers),
    )
    api_transport = httpx.MockTransport(fakes.api.handle)
    monkeypatch.setattr(platform_api, "api_client", lambda: httpx.Client(transport=api_transport))

    def download(reference, directory):
        path = directory / "prompts.jsonl"
        path.write_text("".join(json.dumps({"prompt": p}) + "\n" for p in fakes.prompts))
        return path

    monkeypatch.setattr(distill_main, "download_dataset_version", download)
    monkeypatch.setenv("MLFLOW_RUN_ID", "run-1")
    monkeypatch.setenv("API_URL", "http://api.test:8000")
    monkeypatch.setenv("MLP_STEP_TOKEN", "step-token")
    # main() wraps stdout and stderr; monkeypatch puts the originals back.
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)

    def run(teacher=BASE_MODEL, teacher_url="", gpus="1", **distill):
        distill = {"dataset": "dataset:prompts@1", "teacher": teacher, **distill}
        request = {"name": "qwen-distill", "distill": distill}
        monkeypatch.setattr(sys, "argv", ["mlp-stage", "distill", "7", json.dumps(request)])
        sys.argv += [teacher_url, gpus, str(output)]
        main()

    fakes.run = run
    return fakes


def asked(teacher: FakeTeacher) -> list[dict]:
    return [json.loads(request.content) for request in teacher.requests]


def test_an_in_cluster_teacher_answers_from_vllm_with_its_tool_parser(step):
    step.prompts = ["Weather in Berlin?", "Capital of Italy?"]
    step.teacher.replies = {
        "Weather in Berlin?": calls(("get_weather", {"city": "Berlin"})),
        "Capital of Italy?": text("Rome."),
    }

    step.run(tools=[WEATHER], serving={"tool_parser": "hermes"}, max_tokens=64)

    [vllm] = step.vllm.commands
    assert vllm[:3] == ["vllm", "serve", REPO]
    assert option(vllm, "--tool-call-parser") == "hermes"
    assert "--enable-auto-tool-choice" in vllm
    assert step.vllm.terminated
    first = asked(step.teacher)[0]
    assert str(step.teacher.requests[0].url) == "http://localhost:8000/v1/chat/completions"
    assert first["model"] == "qwen-distill"
    assert first["tools"] == [WEATHER]
    assert first["parallel_tool_calls"] is False
    assert first["max_tokens"] == 64
    assert first["temperature"] == 0.7


def test_replies_are_stored_as_prompt_completion_rows_with_tool_calls_as_dicts(step):
    step.prompts = ["Weather in Berlin?", "Capital of Italy?"]
    step.teacher.replies = {
        "Weather in Berlin?": calls(("get_weather", {"city": "Berlin"})),
        "Capital of Italy?": text("Rome."),
    }

    step.run(tools=[WEATHER], serving={"tool_parser": "hermes"})

    berlin, italy = step.api.rows
    assert berlin["prompt"] == [{"role": "user", "content": "Weather in Berlin?"}]
    [call] = berlin["completion"][0]["tool_calls"]
    assert call == {
        "type": "function",
        "function": {"name": "get_weather", "arguments": {"city": "Berlin"}},
    }
    assert berlin["tools"] == [WEATHER]
    assert italy["completion"] == [{"role": "assistant", "content": "Rome."}]
    [request] = step.api.requests
    assert str(request.url) == "http://api.test:8000/pipelines/7/distill"
    assert request.headers["authorization"] == "Bearer step-token"
    assert step.output.read_text() == "dataset:qwen-distill@1"
    assert step.mlflow.metrics == {"distill/kept": 2, "distill/dropped": 0}


def test_a_prompt_given_as_messages_is_kept_as_they_are(step):
    messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]
    step.prompts = [messages]
    step.teacher.replies = {"Hi": text("Hello.")}

    step.run()

    assert asked(step.teacher)[0]["messages"] == messages
    assert "tools" not in asked(step.teacher)[0]
    [row] = step.api.rows
    assert row["prompt"] == messages
    assert "tools" not in row


MALFORMED = {
    "unknown tool": calls(("get_time", {})),
    "arguments no JSON": calls(("get_weather", "{city: Berlin")),
    "arguments off the schema": calls(("get_weather", {"town": "Berlin"})),
    "several calls": calls(("get_weather", {"city": "A"}), ("get_weather", {"city": "B"})),
    "empty": text(" "),
    "cut off": {**text("The capital of"), "finish_reason": "length"},
}


@pytest.mark.parametrize("malformed", list(MALFORMED))
def test_a_malformed_reply_is_dropped(step, malformed):
    step.prompts = ["bad", *[f"good {n}" for n in range(4)]]
    step.teacher.replies = {"bad": MALFORMED[malformed]} | {
        f"good {n}": text("Fine.") for n in range(4)
    }

    step.run(tools=[WEATHER], serving={"tool_parser": "hermes"})

    assert len(step.api.rows) == 4
    assert step.mlflow.metrics == {"distill/kept": 4, "distill/dropped": 1}
    [dropped] = step.mlflow.artifacts["dropped_replies.json"]
    assert dropped["prompt"] == [{"role": "user", "content": "bad"}]
    assert dropped["reason"]


def test_several_calls_are_kept_when_parallel_calls_are_on(step):
    step.prompts = ["Weather in A and B?"]
    step.teacher.replies = {"Weather in A and B?": MALFORMED["several calls"]}

    step.run(tools=[WEATHER], serving={"tool_parser": "hermes"}, parallel_tool_calls=True)

    assert asked(step.teacher)[0]["parallel_tool_calls"] is True
    assert len(step.api.rows[0]["completion"][0]["tool_calls"]) == 2


def test_more_than_a_fifth_dropped_fails_the_stage_and_registers_nothing(step):
    step.prompts = ["bad 1", "bad 2", "good 1", "good 2", "good 3"]
    step.teacher.replies = {
        "bad 1": text(""),
        "bad 2": text(""),
        **{f"good {n}": text("Fine.") for n in (1, 2, 3)},
    }

    with pytest.raises(SystemExit, match="dropped 2 of 5"):
        step.run()

    assert step.api.requests == []
    assert step.mlflow.metrics == {"distill/kept": 3, "distill/dropped": 2}


def test_an_api_teacher_gets_the_key_which_never_shows(step, monkeypatch, capsys):
    monkeypatch.setenv("MLP_TEACHER_API_KEY", TEACHER_API_KEY)
    step.prompts = ["Hi"]
    step.teacher.status = 401

    with pytest.raises(SystemExit):
        step.run(teacher="gpt-4.1", api_url="https://api.example.com/v1", gpus="0")

    assert step.vllm.commands == []
    [request] = step.teacher.requests
    assert str(request.url) == "https://api.example.com/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {TEACHER_API_KEY}"
    assert json.loads(request.content)["model"] == "gpt-4.1"
    printed = capsys.readouterr()
    assert TEACHER_API_KEY not in printed.out + printed.err
    assert "401" in printed.out
    assert TEACHER_API_KEY not in json.dumps(step.mlflow.artifacts)


def test_an_endpoint_teaches_through_its_service(step):
    step.prompts = ["Hi"]
    step.teacher.replies = {"Hi": text("Hello.")}

    step.run(teacher="endpoint:chat", teacher_url="http://endpoint-chat.mlp.svc:8000", gpus="0")

    assert step.vllm.commands == []
    assert (
        str(step.teacher.requests[0].url) == "http://endpoint-chat.mlp.svc:8000/v1/chat/completions"
    )
    assert asked(step.teacher)[0]["model"] == "chat"
    assert "authorization" not in step.teacher.requests[0].headers
