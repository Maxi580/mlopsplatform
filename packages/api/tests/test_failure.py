import pytest

from mlp_api.pipelines.failure import failure_reason
from mlp_core import config

FREE_MEMORY = (
    "ValueError: Free memory on device cuda:0 (1/8 GiB) on startup is less than desired GPU "
    "memory utilization (0.9, 7 GiB). Decrease GPU memory utilization or reduce GPU memory used "
    "by other processes."
)
ONE_REQUEST = (
    "ValueError: To serve at least one request with the models's max seq len (32768), (8 GiB KV "
    "cache is needed, which is larger than the available KV cache memory (2 GiB)."
)
RELEASED = (
    "AssertionError: Error in memory profiling. Initial free memory 8 GiB, current free memory "
    "9 GiB. This happens when other processes sharing the same container release GPU memory while "
    "vLLM is profiling during initialization."
)


@pytest.mark.parametrize(
    ("log", "message"),
    [
        (
            FREE_MEMORY,
            "Not enough memory left on the GPU right now, including memory other programs use. "
            "Stop something, then Restart.",
        ),
        (
            "RuntimeError: No available memory for the cache blocks. Try increasing "
            "`gpu_memory_utilization` when initializing the engine.",
            "This GPU Share is too small for the model. Restart with a larger one.",
        ),
        (
            ONE_REQUEST,
            "This GPU Share can't hold one request of `max_model_len`. Restart with a larger one "
            "or a lower `max_model_len`.",
        ),
        (
            "    assert init_free_memory >= free_gpu_memory\nAssertionError",
            "Another job released memory while this one was starting. Restart it.",
        ),
        (RELEASED, "Another job released memory while this one was starting. Restart it."),
        (
            "Memory was released during profiling.",
            "Another job released memory while this one was starting. Restart it.",
        ),
    ],
)
def test_a_vllm_memory_error_gets_its_plain_message(log, message):
    assert failure_reason(f"loading\n{log}\n  and a traceback after it", "Restart") == message


@pytest.mark.parametrize(
    "error",
    [
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB",
        "CUDA_ERROR_OUT_OF_MEMORY",
        "Out of GPU memory: tried to allocate 2 GiB",
    ],
)
def test_running_out_of_memory_gets_its_plain_message(error):
    assert failure_reason(error, "Restart") == (
        "Out of GPU memory. Raise the GPU Share, stop something else, or use a smaller model or "
        "batch."
    )


def test_a_pipeline_is_told_to_resume_instead():
    assert failure_reason(FREE_MEMORY, "Resume").endswith("Stop something, then Resume.")
    assert failure_reason(RELEASED, "Resume") == (
        "Another job released memory while this one was starting. Resume it."
    )


def test_a_rewards_memory_limit_in_the_sandbox_doesnt_match():
    log = f"The reward {config.REWARD_FAILURES['memory_limit']}"

    assert "ran out of memory" in log
    assert failure_reason(log, "Resume") == log


def test_a_failure_with_no_match_shows_the_last_log_lines():
    lines = [f"line {index}" for index in range(30)]

    shown = failure_reason("\n".join(lines), "Restart")

    assert shown == "\n".join(lines[-config.FAILURE_SHOWN_LINES :])


def test_no_log_says_so():
    assert failure_reason("", "Restart") == config.NO_LOG_FAILURE
