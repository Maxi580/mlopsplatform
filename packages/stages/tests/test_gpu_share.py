import sys
from types import ModuleType, SimpleNamespace

import pytest

from mlp_core.pipeline_request.schema import Phase
from mlp_stages.gpu_share import cap_gpu_memory

REWARDS = {"short": {"weight": 1.0, "source": "def reward(sample, item):\n    return 1.0\n"}}


@pytest.fixture
def fractions(monkeypatch) -> list[float]:
    """The memory fractions PyTorch is told to keep to."""
    set_fractions = []
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(set_per_process_memory_fraction=set_fractions.append)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return set_fractions


def rl_phase(**settings) -> Phase:
    return Phase(algorithm="grpo", dataset="dataset:rows@1", rewards=REWARDS, settings=settings)


def test_a_step_without_a_gpu_share_is_not_capped(monkeypatch, fractions):
    monkeypatch.delenv("GPU_SHARE", raising=False)

    cap_gpu_memory()

    assert fractions == []


def test_a_step_is_capped_at_its_gpu_share(monkeypatch, fractions):
    monkeypatch.setenv("GPU_SHARE", "0.5")

    cap_gpu_memory()

    assert fractions == [0.5]


# vLLM allocates inside the training process, so the process-wide cap has to hold vLLM's memory too.
def test_colocated_vllm_shares_the_cap_of_its_process(monkeypatch, fractions):
    monkeypatch.setenv("GPU_SHARE", "0.5")

    cap_gpu_memory(rl_phase(vllm_gpu_memory_utilization=0.3))

    assert fractions == [0.5]


@pytest.mark.parametrize("vllm", [0.5, 0.7])
def test_a_share_leaving_training_no_memory_beside_colocated_vllm_fails_the_step(
    monkeypatch, fractions, vllm
):
    monkeypatch.setenv("GPU_SHARE", "0.5")

    with pytest.raises(SystemExit, match="vllm_gpu_memory_utilization"):
        cap_gpu_memory(rl_phase(vllm_gpu_memory_utilization=vllm))

    assert fractions == []


def test_generating_without_vllm_needs_no_memory_for_it(monkeypatch, fractions):
    monkeypatch.setenv("GPU_SHARE", "0.3")

    cap_gpu_memory(rl_phase(use_vllm=False, vllm_gpu_memory_utilization=0.3))

    assert fractions == [0.3]
