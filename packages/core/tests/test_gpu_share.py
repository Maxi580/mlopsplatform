import pytest

from mlp_core.endpoint_spec import ServingOptions
from mlp_core.gpu_share import Gpu, model_memory, training_gpu_share, with_gpu_share
from mlp_core.settings import Settings

GIB = 2**30
GPU_12GB = Gpu(12 * GIB, shared_with_system=False)
GPU_24GB = Gpu(24 * GIB, shared_with_system=False)
GB10 = Gpu(125 * GIB, shared_with_system=True)

# Each model's config.json and weights as they are on Hugging Face.
QWEN_05B = model_memory(
    {
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "hidden_size": 896,
        "max_position_embeddings": 32768,
        "torch_dtype": "bfloat16",
    },
    988097824,
)
LLAMA_8B = model_memory(
    {
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "max_position_embeddings": 8192,
        "torch_dtype": "bfloat16",
    },
    16060522496,
)
LLAMA_70B_W4A16 = model_memory(
    {
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "hidden_size": 8192,
        "head_dim": 128,
        "max_position_embeddings": 131072,
        "torch_dtype": "bfloat16",
    },
    39525311232,
)
QWEN3_32B = model_memory(
    {
        "num_hidden_layers": 64,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "hidden_size": 5120,
        "head_dim": 128,
        "max_position_embeddings": 40960,
        "torch_dtype": "bfloat16",
    },
    65524246528,
)


@pytest.fixture
def settings(platform_settings):
    return Settings(**platform_settings)


def calculated(settings, model, gpu, gpus=1, **options):
    return with_gpu_share(ServingOptions(**options), model, gpu, gpus, settings)


# The table in #116; a model that doesn't fit takes the maximum, in `fp8`. Llama 8B's 25% on the
# GB10 is what the formula gives, where the table says 20%.
@pytest.mark.parametrize(
    ("model", "gpu", "share"),
    [
        (QWEN_05B, GPU_12GB, 0.35),
        (QWEN_05B, GPU_24GB, 0.2),
        (QWEN_05B, GB10, 0.1),
        (LLAMA_8B, GPU_12GB, 0.95),
        (LLAMA_8B, GPU_24GB, 0.85),
        (LLAMA_8B, GB10, 0.25),
        (LLAMA_70B_W4A16, GPU_12GB, 0.95),
        (LLAMA_70B_W4A16, GPU_24GB, 0.95),
        (LLAMA_70B_W4A16, GB10, 0.55),
        (QWEN3_32B, GPU_12GB, 0.95),
        (QWEN3_32B, GPU_24GB, 0.95),
        (QWEN3_32B, GB10, 0.75),
    ],
)
def test_the_share_of_each_model_on_each_gpu(settings, model, gpu, share):
    assert calculated(settings, model, gpu).gpu_memory_utilization == share


def test_max_model_len_is_the_models_limit_up_to_32k(settings):
    assert calculated(settings, LLAMA_8B, GPU_24GB).max_model_len == 8192
    assert calculated(settings, LLAMA_70B_W4A16, GB10).max_model_len == 32768


def test_a_model_without_a_limit_gets_32k(settings):
    config = {"num_hidden_layers": 2, "num_attention_heads": 2, "hidden_size": 64}

    assert calculated(settings, model_memory(config, GIB), GB10).max_model_len == 32768


def test_a_model_that_fits_reserves_the_kv_cache_for_65536_tokens(settings):
    # 1.1 × 15 GiB + 2 GiB + 8 GiB is 26.5 GiB, a little over a quarter of 100 GiB.
    gpu = Gpu(100 * GIB, shared_with_system=False)

    assert calculated(settings, LLAMA_8B, gpu).gpu_memory_utilization == 0.3


def test_first_the_kv_cache_shrinks_to_one_request_of_the_longest_length(settings):
    options = calculated(settings, LLAMA_8B, GPU_24GB)

    assert options.gpu_memory_utilization == 0.85
    assert options.kv_cache_dtype == "auto"


def test_then_the_kv_cache_uses_fp8(settings):
    # 18.5 GiB beside 4 GiB of KV cache is 102% of 22 GiB; beside 2 GiB in `fp8`, 93%.
    options = calculated(settings, LLAMA_8B, Gpu(22 * GIB, False), max_model_len=32768)

    assert options.kv_cache_dtype == "fp8"
    assert options.gpu_memory_utilization == 0.95


def test_then_a_default_takes_the_maximum_in_fp8(settings):
    options = calculated(settings, QWEN3_32B, GPU_12GB)

    assert options.gpu_memory_utilization == 0.95
    assert options.kv_cache_dtype == "fp8"


def test_on_shared_memory_the_maximum_is_lower(settings):
    assert calculated(settings, QWEN3_32B, Gpu(64 * GIB, True)).gpu_memory_utilization == 0.85


def test_a_calculated_share_stays_within_the_gpu_budget(platform_settings):
    settings = Settings(**{**platform_settings, "gpu_budget": 0.5})

    assert calculated(settings, QWEN3_32B, GPU_24GB).gpu_memory_utilization == 0.5


def test_the_gpu_budget_means_nothing_while_sharing_is_off(platform_settings):
    settings = Settings(**{**platform_settings, "gpu_budget": 0.5, "gpu_sharing": False})

    assert calculated(settings, QWEN3_32B, GPU_24GB).gpu_memory_utilization == 0.95


def test_a_users_share_always_wins(settings):
    options = calculated(settings, QWEN3_32B, GPU_12GB, gpu_memory_utilization=1.0)

    assert options.gpu_memory_utilization == 1.0
    assert options.kv_cache_dtype == "auto"


def test_a_users_max_model_len_wins_and_counts(settings):
    options = calculated(settings, LLAMA_70B_W4A16, GB10, max_model_len=131072)

    assert options.max_model_len == 131072
    # 66.5 GiB with 65,536 tokens reserved becomes 86.5 GiB with 131,072.
    assert options.gpu_memory_utilization == 0.7


def test_fp8_halves_the_kv_cache(settings):
    # 18.5 GiB beside a KV cache of 4 GiB, 70.2% of 32 GiB; `auto` takes 8 GiB, 82.7%.
    fp8 = calculated(settings, LLAMA_8B, Gpu(32 * GIB, False), kv_cache_dtype="fp8")
    auto = calculated(settings, LLAMA_8B, Gpu(32 * GIB, False))

    assert fp8.gpu_memory_utilization == 0.75
    assert auto.gpu_memory_utilization == 0.85


def test_a_float32_override_doubles_the_kv_cache(settings):
    # 18.5 GiB beside a KV cache of 16 GiB, 53.8% of 64 GiB; `auto` takes 8 GiB, 41.3%.
    wider = calculated(settings, LLAMA_8B, Gpu(64 * GIB, False), dtype="float32")
    auto = calculated(settings, LLAMA_8B, Gpu(64 * GIB, False))

    assert auto.gpu_memory_utilization == 0.45
    assert wider.gpu_memory_utilization == 0.55


def test_tensor_parallelism_splits_the_weights_and_the_kv_cache(settings):
    one = calculated(settings, LLAMA_8B, Gpu(64 * GIB, False))
    two = calculated(settings, LLAMA_8B, Gpu(64 * GIB, False), gpus=2)

    assert one.gpu_memory_utilization == 0.45
    assert two.gpu_memory_utilization == 0.25


def test_a_config_inside_text_config_counts(settings):
    nested = {
        "text_config": {
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "max_position_embeddings": 8192,
        }
    }

    assert model_memory(nested, 16060522496) == LLAMA_8B


def test_a_config_that_doesnt_say_gives_no_model_memory():
    assert model_memory({"model_type": "tiny"}, GIB) is None


def test_an_unknown_model_keeps_the_fixed_share_and_vllms_own_length(settings):
    options = calculated(settings, None, GB10)

    assert options.gpu_memory_utilization == 0.3
    assert options.max_model_len is None


def test_an_unknown_gpu_keeps_the_fixed_share_but_passes_the_length(settings):
    options = calculated(settings, LLAMA_8B, None)

    assert options.gpu_memory_utilization == 0.3
    assert options.max_model_len == 8192


@pytest.mark.parametrize(
    ("method", "gpu", "share"),
    [
        # 2 × 15 GiB + 2 GiB and 6 × 15 GiB + 2 GiB, of 125 GiB.
        ("lora", GB10, 0.3),
        ("qlora", GB10, 0.3),
        ("full", GB10, 0.75),
        # Too high only makes a job wait, and 92 GiB can't fit on 24 GiB, so the maximum.
        ("full", GPU_24GB, 0.95),
    ],
)
def test_the_share_of_training(settings, method, gpu, share):
    assert training_gpu_share(16060522496, method, 0, 0, gpu, settings) == share


def test_a_teacher_trained_against_counts_once(settings):
    # 32 GiB of training beside 36.8 GiB of Teacher weights, of 125 GiB.
    assert training_gpu_share(16060522496, "lora", 39525311232, 0, GB10, settings) == 0.55


def test_colocated_vllm_takes_its_share_on_top_of_training(settings):
    # 25.5% to train, beside the 30% vLLM takes out of the same share.
    assert training_gpu_share(16060522496, "lora", 0, 0.3, GB10, settings) == 0.6


def test_colocated_vllm_stays_within_the_maximum(settings):
    assert training_gpu_share(16060522496, "lora", 0, 0.8, GB10, settings) == 0.85
