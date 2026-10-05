import json
import subprocess
import sys

import httpx2 as httpx
import pytest

from mlp_core.pipeline_request.schema import Reward
from mlp_stages import sandbox
from mlp_stages.finetune.rewards import reward_functions

CORRECT = "def reward(sample, item):\n    return float(item['answer'] in sample['output_text'])"
TIMEOUT_SECONDS = 2


class LocalSandbox:
    """Runs each snippet as the Sandbox does, in a fresh Python process, without its isolation."""

    def __init__(self):
        self.batches = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        snippets = json.loads(request.content)
        self.batches.append(snippets)
        return httpx.Response(200, json=[self.run(snippet) for snippet in snippets])

    def run(self, snippet: dict) -> dict:
        try:
            done = subprocess.run(
                [sys.executable, "-I", "-c", snippet["code"]],
                input=snippet["input"],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "stdout": "", "stderr": ""}
        status = "ok" if done.returncode == 0 else "error"
        return {"status": status, "stdout": done.stdout, "stderr": done.stderr}


@pytest.fixture
def local_sandbox(monkeypatch):
    local = LocalSandbox()
    monkeypatch.setenv("SANDBOX_URL", "http://sandbox.test:8090")
    monkeypatch.setattr(
        sandbox, "sandbox_client", lambda: httpx.Client(transport=httpx.MockTransport(local.handle))
    )
    return local


class Metrics(dict):
    def __call__(self, name: str, value: float) -> None:
        self[name] = value


def score(source: str, completions: list, name="correct", **columns) -> tuple[list, Metrics]:
    """The rewards of the completions, as TRL asks for them, and the metrics logged."""
    [function] = reward_functions({name: Reward(weight=1.0, source=source)})
    metrics = Metrics()
    prompts = ["What is 2 + 2?"] * len(completions)
    columns = columns or {"answer": ["4"] * len(completions)}
    rewards = function(
        prompts=prompts,
        completions=completions,
        completion_ids=[[1]] * len(completions),
        trainer_state=object(),
        log_extra=lambda *args: None,
        log_metric=metrics,
        **columns,
    )
    return rewards, metrics


def test_each_completion_is_scored_against_its_dataset_row(local_sandbox):
    rewards, metrics = score(CORRECT, ["It is 4.", "It is 5."])

    assert rewards == [1.0, 0.0]
    assert metrics == {"rewards/correct/errors": 0}


def test_a_conversation_is_scored_by_its_reply_and_tool_calls(local_sandbox):
    source = (
        "def reward(sample, item):\n"
        "    [call] = sample['output_tools']\n"
        "    return float(call['function']['name'] == item['tool'] and sample['output_text'] == '')"
    )
    call = {"type": "function", "function": {"name": "get_weather", "arguments": {}}}
    reply = [{"role": "assistant", "content": "", "tool_calls": [call]}]

    rewards, _ = score(source, [reply], tool=["get_weather"])

    assert rewards == [1.0]


def test_the_item_holds_the_prompt_too(local_sandbox):
    source = "def reward(sample, item):\n    return float(item['prompt'].endswith('?'))"

    assert score(source, ["4"])[0] == [1.0]


def test_none_means_the_reward_does_not_apply(local_sandbox):
    rewards, _ = score("def reward(sample, item):\n    return None", ["4"])

    assert rewards == [None]


def test_what_a_reward_prints_does_not_spoil_its_score(local_sandbox):
    source = "def reward(sample, item):\n    print('checking', 0.5)\n    return 0.25"

    assert score(source, ["4"])[0] == [0.25]


def test_a_batch_of_completions_goes_to_the_sandbox_at_once(local_sandbox):
    score(CORRECT, ["4", "5", "6"])

    [batch] = local_sandbox.batches
    assert len(batch) == 3


@pytest.mark.parametrize(
    ("source", "reported"),
    [
        ("def reward(sample, item):\n    return 1 / 0", "ZeroDivisionError: division by zero"),
        ("def reward(sample, item):\n    return 'yes'", "returned 'yes', not a number or None"),
        ("def reward(sample, item):\n    while True: pass", "timed out"),
        (
            "import sys\ndef reward(sample, item):\n    sys.exit()",
            "exited without returning a score",
        ),
    ],
)
def test_a_reward_that_fails_scores_zero_and_is_reported(local_sandbox, capsys, source, reported):
    rewards, metrics = score(source, ["It is 4.", "It is 5."])

    assert rewards == [0.0, 0.0]
    assert metrics == {"rewards/correct/errors": 2}
    assert f"Reward `correct` failed on 2 of 2 completions; the first: {reported}" in (
        capsys.readouterr().err
    )


def test_each_function_is_named_after_its_reward_for_trls_metrics(local_sandbox):
    rewards = {
        "correct": Reward(weight=0.8, source=CORRECT),
        "short": Reward(weight=0.2, source=CORRECT),
    }

    assert [function.__name__ for function in reward_functions(rewards)] == ["correct", "short"]


def test_on_trl_before_1_0_errors_are_logged_into_the_active_run(local_sandbox, monkeypatch):
    # Unsloth trains on TRL 0.24, which hands reward functions no log_metric.
    mlflow = pytest.importorskip("mlflow", reason="needs the hf or unsloth extra")

    logged = []
    monkeypatch.setattr(mlflow, "log_metric", lambda *args, **kwargs: logged.append((args, kwargs)))
    [function] = reward_functions({"correct": Reward(weight=1.0, source=CORRECT)})

    rewards = function(
        prompts=["What is 2 + 2?"],
        completions=["It is 5."],
        completion_ids=[[1]],
        trainer_state=type("TrainerState", (), {"global_step": 7})(),
        answer=["4"],
    )

    assert rewards == [0.0]
    assert logged == [(("rewards/correct/errors", 0), {"step": 7})]
