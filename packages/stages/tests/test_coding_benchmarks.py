import importlib.util
import json
from pathlib import Path

import httpx2 as httpx
import pytest

lm_eval = pytest.importorskip(
    "lm_eval", reason="needs the evaluate extra: uv sync --extra evaluate"
)

import evaluate  # noqa: E402
from evalscope.api.dataset import Sample  # noqa: E402
from evalscope.api.evaluator import TaskState  # noqa: E402
from evalscope.api.mixin import CodeExecutionSandboxMixin  # noqa: E402
from evalscope.api.registry import get_benchmark  # noqa: E402
from evalscope.config import TaskConfig  # noqa: E402

from mlp_stages import sandbox  # noqa: E402
from mlp_stages.evaluate import evalscope_harness, lm_eval_harness  # noqa: E402

ADD = "def add(a, b):\n    return a + b"
ADD_TEST = "assert add(2, 3) == 5"


class FakeSandbox:
    """Answers each snippet with the status set for its code, else `error`; runs nothing."""

    def __init__(self):
        self.batches = []
        self.statuses = {}

    def handle(self, request: httpx.Request) -> httpx.Response:
        snippets = json.loads(request.content)
        self.batches.append([snippet["code"] for snippet in snippets])
        results = [
            {"status": self.statuses.get(snippet["code"], "error"), "stdout": "4\n", "stderr": ""}
            for snippet in snippets
        ]
        return httpx.Response(200, json=results)


@pytest.fixture
def fake_sandbox(monkeypatch):
    fake = FakeSandbox()
    monkeypatch.setenv("SANDBOX_URL", "http://sandbox.test:8090")
    monkeypatch.setattr(
        sandbox, "sandbox_client", lambda: httpx.Client(transport=httpx.MockTransport(fake.handle))
    )
    # The shims patch these; monkeypatch puts the originals back.
    monkeypatch.setattr(evaluate, "load", evaluate.load)
    for name in ("use_sandbox", "execute_code_in_sandbox"):
        monkeypatch.setattr(CodeExecutionSandboxMixin, name, vars(CodeExecutionSandboxMixin)[name])
    return fake


def lm_eval_task_utils(task: str):
    """The `utils.py` of an lm-eval task, imported as lm-eval does when it loads the task."""
    path = Path(lm_eval.__file__).parent / "tasks" / task / "utils.py"
    spec = importlib.util.spec_from_file_location(f"{task}_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_humaneval_runs_each_program_and_its_tests_in_the_sandbox(fake_sandbox):
    lm_eval_harness.score_code_in_sandbox()
    utils = lm_eval_task_utils("humaneval")
    fake_sandbox.statuses[f"{ADD}\n{ADD_TEST}"] = "ok"

    score = utils.pass_at_k(references=[ADD_TEST], predictions=[[ADD]], k=[1])

    assert score == {"pass@1": 1.0}
    assert fake_sandbox.batches[-1] == [f"{ADD}\n{ADD_TEST}"]


def test_mbpp_scores_a_program_that_fails_its_tests_as_zero(fake_sandbox):
    lm_eval_harness.score_code_in_sandbox()
    utils = lm_eval_task_utils("mbpp")

    assert utils.pass_at_1(references=ADD_TEST, predictions=["def add(a, b): pass"]) == 0.0
    assert fake_sandbox.batches[-1] == [f"def add(a, b): pass\n{ADD_TEST}"]


def test_pass_at_k_counts_the_candidates_that_passed_from_one_batch(fake_sandbox):
    lm_eval_harness.score_code_in_sandbox()
    fake_sandbox.statuses[f"{ADD}\n{ADD_TEST}"] = "ok"
    code_eval = evaluate.load("code_eval")

    score, _ = code_eval.compute(
        references=[ADD_TEST, ADD_TEST], predictions=[[ADD, "pass"], ["pass", "pass"]], k=[1, 2, 3]
    )

    assert score == {"pass@1": 0.25, "pass@2": 0.5}
    assert len(fake_sandbox.batches) == 1


def test_evalscope_scores_a_coding_sample_by_running_it_in_the_sandbox(fake_sandbox):
    evalscope_harness.score_code_in_sandbox()
    config = TaskConfig(model="qwen", eval_type="mock_llm", datasets=["mbpp_plus"])
    adapter = get_benchmark("mbpp_plus", config=config)
    sample = Sample(input="Add two numbers.", metadata={"task_id": 1, "test_list": [ADD_TEST]})
    program = f"{ADD}\n{ADD_TEST}\n"
    fake_sandbox.statuses[program] = "ok"

    score = adapter.match_score(ADD, ADD, "", TaskState(model="qwen", sample=sample))

    assert score.value == {"acc": True}
    assert fake_sandbox.batches == [[program]]


def test_evalscope_reads_what_a_program_printed_in_the_sandbox(fake_sandbox):
    evalscope_harness.score_code_in_sandbox()
    adapter = get_benchmark("mbpp_plus", config=TaskConfig(model="qwen", eval_type="mock_llm"))
    fake_sandbox.statuses["print(2 + 2)"] = "ok"

    result = adapter.execute_code_in_sandbox("print(2 + 2)", timeout=6, language="python")

    assert result["status"] == "success"
    assert result["output"] == "4\n"
    assert adapter.execute_code_in_sandbox("echo 4", language="shell")["status"] == "error"
    assert fake_sandbox.batches == [["print(2 + 2)"]]
