import pytest

from mlp_core.endpoint_spec import ServingOptions
from mlp_core.gpu_share import (
    Gpu,
    ModelMemory,
    model_length_limit,
    training_gpu_share,
    with_gpu_share,
)
from mlp_core.settings import Settings

GIB = 2**30
GPU_24GB = Gpu(24 * GIB, shared_with_system=False)
GB10 = Gpu(125 * GIB, shared_with_system=True)
LLAMA_8B_BYTES = 15 * GIB


@pytest.fixture
def settings(platform_settings):
    return Settings(**platform_settings)


def calculated(settings, weights=LLAMA_8B_BYTES, gpu=GPU_24GB, gpus=1, limit=None, **options):
    model = ModelMemory(weights, limit)
    return with_gpu_share(ServingOptions(**options), model, gpu, gpus, settings)


def test_vllm_needs_one_and_a_half_times_its_weights_and_a_reserve(settings):
    # 22.5 GiB + 2 GiB of 50 GiB is 49%, rounded up to a slice.
    gpu = Gpu(50 * GIB, shared_with_system=False)

    assert calculated(settings, gpu=gpu).gpu_memory_utilization == 0.5


def test_shared_memory_adds_its_reserve(settings):
    # 22.5 GiB + 2 GiB + 4 GiB of 125 GiB is 22.8%; without the 4 GiB it would be 19.6%.
    assert calculated(settings, gpu=GB10).gpu_memory_utilization == 0.25
    unshared = Gpu(125 * GIB, shared_with_system=False)
    assert calculated(settings, gpu=unshared).gpu_memory_utilization == 0.2


def test_weights_on_several_gpus_are_divided_but_the_reserve_is_not(settings):
    # 11.25 GiB + 2 GiB of 50 GiB is 26.5%.
    gpu = Gpu(50 * GIB, shared_with_system=False)

    assert calculated(settings, gpu=gpu, gpus=2).gpu_memory_utilization == 0.3


def test_a_calculated_share_stops_at_the_recommended_maximum(settings):
    assert calculated(settings, weights=64 * GIB).gpu_memory_utilization == 0.85


def test_a_share_exactly_on_a_slice_stays_there(settings):
    # 1.5 × 2 GiB + 2 GiB of 20 GiB is exactly 25%.
    gpu = Gpu(20 * GIB, shared_with_system=False)

    assert calculated(settings, weights=2 * GIB, gpu=gpu).gpu_memory_utilization == 0.25


def test_a_users_share_always_wins_above_the_maximum_too(settings):
    options = calculated(settings, weights=64 * GIB, gpu_memory_utilization=1.0)

    assert options.gpu_memory_utilization == 1.0


@pytest.mark.parametrize(("weights", "gpu"), [(None, GPU_24GB), (LLAMA_8B_BYTES, None)])
def test_unknown_weights_or_gpu_memory_mean_the_fixed_share(settings, weights, gpu):
    assert calculated(settings, weights=weights, gpu=gpu).gpu_memory_utilization == 0.3


def test_the_length_is_the_models_limit_up_to_32k(settings):
    assert calculated(settings, limit=8192).max_model_len == 8192
    assert calculated(settings, limit=131072).max_model_len == 32768


def test_a_model_whose_limit_cant_be_read_passes_no_length(settings):
    assert calculated(settings).max_model_len is None


def test_a_users_length_wins_over_the_models_limit(settings):
    assert calculated(settings, limit=8192, max_model_len=131072).max_model_len == 131072
    assert calculated(settings, max_model_len=4096).max_model_len == 4096


def test_the_length_is_read_even_where_the_weights_arent(settings):
    options = calculated(settings, weights=None, limit=8192)

    assert (options.gpu_memory_utilization, options.max_model_len) == (0.3, 8192)


def test_the_lowest_limit_of_a_config_counts():
    assert model_length_limit({"max_position_embeddings": 8192, "n_positions": 4096}) == 4096


def test_a_limit_inside_text_config_counts():
    assert model_length_limit({"text_config": {"max_position_embeddings": 8192}}) == 8192


def test_a_config_without_a_limit_gives_none():
    assert model_length_limit({"model_type": "tiny"}) is None


@pytest.mark.parametrize(
    ("method", "share"),
    [
        # 2 × 15 GiB + 2 GiB and 6 × 15 GiB + 2 GiB, of 125 GiB.
        ("lora", 0.3),
        ("qlora", 0.3),
        ("full", 0.75),
        (None, 0.3),
    ],
)
def test_the_share_of_training(settings, method, share):
    assert training_gpu_share(LLAMA_8B_BYTES, method, 0, 0, GB10, settings) == share


def test_training_stops_at_the_recommended_maximum(settings):
    assert training_gpu_share(LLAMA_8B_BYTES, "full", 0, 0, GPU_24GB, settings) == 0.85


def test_a_teacher_counts_with_its_weights_once(settings):
    # 32 GiB of training beside 36 GiB of Teacher weights, of 125 GiB.
    assert training_gpu_share(LLAMA_8B_BYTES, "lora", 36 * GIB, 0, GB10, settings) == 0.55


def test_colocated_vllm_takes_its_share_on_top_of_training(settings):
    # 25.6% to train, beside the 30% vLLM takes out of the same share.
    assert training_gpu_share(LLAMA_8B_BYTES, "lora", 0, 0.3, GB10, settings) == 0.6
    assert training_gpu_share(LLAMA_8B_BYTES, "lora", 0, 0.8, GB10, settings) == 0.85
