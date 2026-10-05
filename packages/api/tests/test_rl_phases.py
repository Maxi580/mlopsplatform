import pytest

from .test_datasets import jsonl, upload
from .test_evaluate_stage import container, errors, validate
from .test_pipeline_request import pipeline_request
from .test_pipelines import submit, submittable  # noqa: F401

pytestmark = pytest.mark.usefixtures("submittable", "maths")

CORRECT = "def reward(sample, item):\n    return float(item['answer'] in sample['output_text'])"
SHORT = "def reward(sample, item):\n    return 1.0 if len(sample['output_text']) < 200 else 0.0"


@pytest.fixture
def maths(logged_in_api):
    """The `maths` Dataset: prompts, each with the answer a reward checks."""
    upload(logged_in_api, "maths", jsonl({"prompt": "What is 2 + 2?", "answer": "4"}))


def reinforcing(algorithm="grpo", **phase) -> dict:
    """A request whose one Phase learns from two weighted rewards on `maths`."""
    settings = {
        "learning_rate": 1e-6,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "max_completion_length": 256,
    }
    rewards = {
        "correct": {"weight": 0.8, "source": CORRECT},
        "short": {"weight": 0.2, "source": SHORT},
    }
    rl = {"algorithm": algorithm, "dataset": "dataset:maths", "rewards": rewards}
    return pipeline_request(phase={**rl, "settings": settings, **phase})


@pytest.mark.parametrize("algorithm", ["grpo", "rloo"])
def test_an_rl_phase_with_weighted_rewards_is_submitted(logged_in_api, algorithm):
    response = validate(logged_in_api, reinforcing(algorithm))

    assert response.status_code == 200, response.text
    phase = response.json()["request"]["finetune"]["phases"][0]
    assert phase["dataset"] == "dataset:maths@1"
    assert list(phase["rewards"]) == ["correct", "short"]
    assert submit(logged_in_api, reinforcing(algorithm)).status_code == 202


@pytest.mark.parametrize("algorithm", ["grpo", "rloo"])
@pytest.mark.parametrize("rewards", [None, {}])
def test_an_rl_phase_without_rewards_is_rejected(logged_in_api, algorithm, rewards):
    [error] = errors(logged_in_api, reinforcing(algorithm, rewards=rewards))

    assert error["loc"] == ["finetune", "phases", 0]
    assert f"a {algorithm} Phase learns from `rewards`; name at least one" in error["msg"]


def test_only_an_rl_phase_takes_rewards(logged_in_api):
    rewards = {"correct": {"weight": 1, "source": CORRECT}}

    [error] = errors(logged_in_api, pipeline_request(phase={"rewards": rewards}))

    assert "a sft Phase has no `rewards`" in error["msg"]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("def reward(sample, item)\n    return 1.0", "is no Python: expected ':' (line 1)"),
        ("def score(sample, item):\n    return 1.0", "defines no `def reward(sample, item)`"),
        ("def reward(sample):\n    return 1.0", "defines no `def reward(sample, item)`"),
    ],
)
def test_a_reward_that_cannot_run_is_rejected_at_submit(logged_in_api, source, message):
    rewards = {"correct": {"weight": 1, "source": source}}

    [error] = errors(logged_in_api, reinforcing(rewards=rewards))

    assert error["loc"] == ["finetune", "phases", 0, "rewards", "correct", "source"]
    assert error["msg"] == f"`source` {message}"


def test_a_reward_name_must_name_a_metric(logged_in_api):
    rewards = {"is correct?": {"weight": 1, "source": CORRECT}}

    [error] = errors(logged_in_api, reinforcing(rewards=rewards))

    assert error["loc"][:4] == ["finetune", "phases", 0, "rewards"]


def test_a_reward_source_is_capped(logged_in_api):
    source = CORRECT + "\n#" + "x" * 2**16
    rewards = {"correct": {"weight": 1, "source": source}}

    [error] = errors(logged_in_api, reinforcing(rewards=rewards))

    assert error["loc"] == ["finetune", "phases", 0, "rewards", "correct", "source"]


@pytest.mark.parametrize(
    "setting", ["vllm_mode", "vllm_server_base_url", "reward_weights", "reward_funcs"]
)
def test_the_platform_owns_rollouts_and_reward_weights(logged_in_api, setting):
    settings = {**reinforcing()["finetune"]["phases"][0]["settings"], setting: "x"}

    [error] = errors(logged_in_api, reinforcing(settings=settings))

    assert error["loc"][-1] == setting


def test_settings_are_checked_against_the_rloo_config(logged_in_api):
    settings = {**reinforcing()["finetune"]["phases"][0]["settings"], "max_length": 512}

    [error] = errors(logged_in_api, reinforcing("rloo", settings=settings))

    assert error["msg"] == "`max_length` is not a RLOOConfig setting"


def test_an_rl_phase_trains_on_prompts(logged_in_api):
    [error] = errors(logged_in_api, reinforcing(dataset="dataset:chat"))

    assert "grpo trains on prompt_only rows" in error["msg"]


def test_the_finetune_step_reaches_the_sandbox(logged_in_api, cluster):
    assert submit(logged_in_api, reinforcing()).status_code == 202

    env = {
        variable["name"]: variable["value"] for variable in container(cluster, "finetune")["env"]
    }
    assert env["SANDBOX_URL"]
