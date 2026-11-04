import json

import pytest
from sqlalchemy import update

from mlp_api.endpoints.table import endpoint
from mlp_api.pipelines.hugging_face import HubModel
from mlp_core import api_paths, config

from .test_datasets import jsonl, upload
from .test_endpoints import contains, pod, register_adapter, restart, start, stop, vllm_args
from .test_judge import save
from .test_pipeline_request import BASE_MODEL, COMMIT, pipeline_request
from .test_pipelines import submit, submittable, validate  # noqa: F401

GIB = 2**30
LLAMA = "meta-llama/Llama-3-8B"
PINNED_LLAMA = f"hf:{LLAMA}@{COMMIT}"
LLAMA_CONFIG = {
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "hidden_size": 4096,
    "max_position_embeddings": 8192,
    "torch_dtype": "bfloat16",
}
LLAMA_BYTES = 16060522496
# A KV cache of a few KiB, so an Adapter's bytes can tip a share over a slice.
TINY_CONFIG = {"num_hidden_layers": 1, "num_attention_heads": 1, "hidden_size": 1}


def on_the_hub(hugging_face, repo, weights_bytes, model_config):
    hugging_face.models[repo] = HubModel(commit=COMMIT, needs_remote_code=False)
    hugging_face.files[repo] = {
        "config.json": json.dumps(model_config).encode(),
        config.WEIGHTS_INDEX_FILE: json.dumps({"metadata": {"total_size": weights_bytes}}).encode(),
    }


def with_gpu(cluster, gib, shared_with_system=False):
    reading = {"memory_total_bytes": int(gib * GIB), "shared_with_system": shared_with_system}
    cluster.gpu_readings = {"host": [reading]}


@pytest.fixture
def llama(logged_in_api, hugging_face):
    on_the_hub(hugging_face, LLAMA, LLAMA_BYTES, LLAMA_CONFIG)


def listed(api, name="chat"):
    return next(e for e in api.get(api_paths.ENDPOINTS).json() if e["name"] == name)


def share_of(api, name="chat"):
    return listed(api, name)["gpu_share"]


def test_an_endpoint_runs_on_the_share_calculated_for_its_model_and_gpu(
    logged_in_api, llama, cluster
):
    with_gpu(cluster, 24)

    assert start(logged_in_api, PINNED_LLAMA).status_code == 201

    assert share_of(logged_in_api) == 0.85
    assert contains(vllm_args(cluster), ["--gpu-memory-utilization", "0.85"])
    assert contains(vllm_args(cluster), ["--max-model-len", "8192"])
    [vllm] = pod(cluster)["containers"]
    assert vllm["resources"]["limits"] == {"nvidia.com/gpu": 17}


def test_a_shared_memory_gpu_counts_its_reserve(logged_in_api, llama, cluster):
    with_gpu(cluster, 125, shared_with_system=True)

    start(logged_in_api, PINNED_LLAMA)

    assert share_of(logged_in_api) == 0.25


def test_an_endpoint_that_barely_fits_is_served_in_fp8(logged_in_api, llama, cluster):
    with_gpu(cluster, 20)

    start(logged_in_api, PINNED_LLAMA)

    assert share_of(logged_in_api) == 0.95
    assert contains(vllm_args(cluster), ["--kv-cache-dtype", "fp8"])


def test_a_users_share_always_wins(logged_in_api, llama, cluster):
    with_gpu(cluster, 12)

    start(logged_in_api, PINNED_LLAMA, gpu_memory_utilization=1.0)

    assert share_of(logged_in_api) == 1.0
    assert not contains(vllm_args(cluster), ["--kv-cache-dtype", "fp8"])


def test_a_calculated_share_above_the_budget_runs_at_the_budget(logged_in_api, llama, cluster):
    logged_in_api.app.state.settings.gpu_budget = 0.5
    with_gpu(cluster, 24)

    assert start(logged_in_api, PINNED_LLAMA).status_code == 201

    assert share_of(logged_in_api) == 0.5


def test_a_restart_keeps_the_share_even_on_another_gpu(logged_in_api, llama, cluster):
    with_gpu(cluster, 24)
    start(logged_in_api, PINNED_LLAMA)
    stop(logged_in_api)

    with_gpu(cluster, 125, shared_with_system=True)
    assert restart(logged_in_api).status_code == 200

    assert share_of(logged_in_api) == 0.85
    assert contains(vllm_args(cluster), ["--gpu-memory-utilization", "0.85"])


def test_a_model_without_an_index_takes_its_one_weights_file(logged_in_api, hugging_face, cluster):
    on_the_hub(hugging_face, LLAMA, 0, LLAMA_CONFIG)
    del hugging_face.files[LLAMA][config.WEIGHTS_INDEX_FILE]
    hugging_face.file_sizes[LLAMA] = {config.WEIGHTS_FILE: LLAMA_BYTES}
    with_gpu(cluster, 24)

    start(logged_in_api, PINNED_LLAMA)

    assert share_of(logged_in_api) == 0.85


def test_a_model_without_weights_to_measure_keeps_the_fixed_share(
    logged_in_api, hugging_face, cluster
):
    on_the_hub(hugging_face, LLAMA, 0, LLAMA_CONFIG)
    del hugging_face.files[LLAMA][config.WEIGHTS_INDEX_FILE]
    with_gpu(cluster, 24)

    start(logged_in_api, PINNED_LLAMA)

    assert share_of(logged_in_api) == 0.3
    assert not contains(vllm_args(cluster), ["--max-model-len", "8192"])


def test_an_unknown_gpu_size_keeps_the_fixed_share_but_passes_the_length(
    logged_in_api, llama, cluster
):
    start(logged_in_api, PINNED_LLAMA)

    assert share_of(logged_in_api) == 0.3
    assert contains(vllm_args(cluster), ["--max-model-len", "8192"])


def test_a_failing_gpu_reporter_keeps_the_fixed_share(logged_in_api, llama, cluster, monkeypatch):
    def fail():
        raise ConnectionError("the Kubernetes API is down")

    monkeypatch.setattr(cluster, "machine_gpu_readings", fail)

    assert start(logged_in_api, PINNED_LLAMA).status_code == 201

    assert share_of(logged_in_api) == 0.3


def test_an_adapter_adds_its_file_to_its_base(
    logged_in_api, hugging_face, model_registry, object_store, cluster
):
    # 2 GiB + 1.1 MiB + about 0.3 MiB of KV cache just fits 80%; the Adapter's 1.1 MiB doesn't.
    on_the_hub(hugging_face, LLAMA, 2**20, TINY_CONFIG)
    with_gpu(cluster, (2 + 2 / 1024) / 0.8)
    prefix = register_adapter(model_registry, object_store, "sft", PINNED_LLAMA)
    object_store.buckets["mlflow"][f"{prefix}adapter_model.safetensors"] = b"a" * 2**20

    start(logged_in_api, PINNED_LLAMA, name="base")
    start(logged_in_api, "model:sft@1", name="adapter")

    assert share_of(logged_in_api, "base") == 0.8
    assert share_of(logged_in_api, "adapter") == 0.85


def test_a_quantized_model_counts_its_quantized_size(logged_in_api, hugging_face, cluster):
    llama_70b_w4a16 = {
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "hidden_size": 8192,
        "head_dim": 128,
        "max_position_embeddings": 131072,
        "quantization_config": {"quant_method": "compressed-tensors"},
    }
    on_the_hub(hugging_face, LLAMA, 39525311232, llama_70b_w4a16)
    with_gpu(cluster, 125, shared_with_system=True)

    start(logged_in_api, PINNED_LLAMA)

    assert share_of(logged_in_api) == 0.55
    assert contains(vllm_args(cluster), ["--max-model-len", "32768"])


def test_the_judge_runs_on_a_calculated_share(logged_in_api, llama, cluster):
    with_gpu(cluster, 24)

    assert save(logged_in_api, spec={"model": PINNED_LLAMA}).status_code == 200

    assert share_of(logged_in_api, config.JUDGE_ENDPOINT_NAME) == 0.85


def test_the_preview_says_what_a_start_would_pin_and_starts_nothing(logged_in_api, llama, cluster):
    with_gpu(cluster, 24)

    response = logged_in_api.post(api_paths.PREVIEW_ENDPOINT, json={"model": f"hf:{LLAMA}"})

    assert response.status_code == 200, response.text
    assert response.json()["model"] == PINNED_LLAMA
    assert response.json()["gpu_memory_utilization"] == 0.85
    assert response.json()["max_model_len"] == 8192
    assert cluster.endpoints == {}
    assert logged_in_api.get(api_paths.ENDPOINTS).json() == []


def test_the_preview_refuses_what_a_start_would(logged_in_api, llama, cluster):
    logged_in_api.app.state.settings.gpu_budget = 0.5
    answer = logged_in_api.post(
        api_paths.PREVIEW_ENDPOINT, json={"model": PINNED_LLAMA, "gpu_memory_utilization": 0.6}
    )
    unknown = logged_in_api.post(api_paths.PREVIEW_ENDPOINT, json={"model": "hf:nobody/nothing"})

    assert answer.status_code == 422
    assert "can never fit" in answer.json()["detail"]
    assert unknown.status_code == 422


def test_an_endpoint_stored_before_calculated_shares_shows_the_then_default(
    logged_in_api, llama, cluster
):
    # As a row from before: the default left out, and no `max_model_len`.
    start(logged_in_api, PINNED_LLAMA)
    assert listed(logged_in_api)["spec"]["gpu_memory_utilization"] == 0.3
    with logged_in_api.app.state.engine.begin() as connection:
        connection.execute(update(endpoint).values(spec={"model": PINNED_LLAMA}))
    stop(logged_in_api)
    with_gpu(cluster, 24)

    assert restart(logged_in_api).status_code == 200

    assert share_of(logged_in_api) == 0.3
    assert contains(vllm_args(cluster), ["--gpu-memory-utilization", "0.3"])
    assert not contains(vllm_args(cluster), ["--max-model-len", "8192"])


@pytest.fixture
def base_model_files(logged_in_api, hugging_face, submittable):  # noqa: F811
    on_the_hub(hugging_face, BASE_MODEL, LLAMA_BYTES, LLAMA_CONFIG)


def on_prompts(api) -> dict:
    """What a Phase that reads prompt rows needs, as those rows have no `max_length` to cut to."""
    upload(api, "prompts", jsonl({"prompt": "Why is the sky blue?"}))
    return {"dataset": "dataset:prompts", "settings": {"max_completion_length": 64}}


def validated(api, request):
    response = validate(api, request)
    assert response.status_code == 200, response.text
    return response.json()


def test_a_pipelines_share_is_calculated_and_shown_before_it_is_submitted(
    logged_in_api, base_model_files, cluster
):
    with_gpu(cluster, 125, shared_with_system=True)

    answer = validated(logged_in_api, pipeline_request())

    assert answer["request"]["gpu_share"] == 0.3
    assert answer["gpu_shares"] == {"finetune": 0.3}


def test_a_full_phase_needs_more_than_an_adapter(logged_in_api, base_model_files, cluster):
    with_gpu(cluster, 125, shared_with_system=True)
    request = pipeline_request(phase={"method": "full", "lora": None})

    assert validated(logged_in_api, request)["request"]["gpu_share"] == 0.75


def test_the_largest_step_decides_the_pipelines_share(logged_in_api, base_model_files, cluster):
    with_gpu(cluster, 125, shared_with_system=True)
    request = pipeline_request()
    request["finetune"]["phases"].append(
        {
            "algorithm": "sft",
            "dataset": "dataset:chat",
            "method": "full",
            "settings": {"max_length": 64},
        }
    )
    request["finetune"]["phases"][0]["output"] = "merged"

    assert validated(logged_in_api, request)["request"]["gpu_share"] == 0.75


def test_a_users_gpu_share_wins(logged_in_api, base_model_files, cluster):
    with_gpu(cluster, 125, shared_with_system=True)
    request = {**pipeline_request(), "gpu_share": 0.4}

    assert validated(logged_in_api, request)["gpu_shares"] == {"finetune": 0.4}


def test_an_unknown_gpu_keeps_the_fixed_pipeline_share(logged_in_api, base_model_files):
    assert validated(logged_in_api, pipeline_request())["request"]["gpu_share"] == 0.5


def test_an_unknown_model_keeps_the_fixed_pipeline_share(logged_in_api, submittable, cluster):  # noqa: F811
    with_gpu(cluster, 24)

    assert validated(logged_in_api, pipeline_request())["request"]["gpu_share"] == 0.5


def test_evaluates_own_vllm_is_sized_for_the_model_behind_its_output(
    logged_in_api, base_model_files, cluster
):
    with_gpu(cluster, 125, shared_with_system=True)
    request = {**pipeline_request(), "evaluate": {"benchmarks": ["lm_eval:gsm8k"]}}

    answer = validated(logged_in_api, request)

    serving = answer["request"]["evaluate"]["serving"]
    assert (serving["gpu_memory_utilization"], serving["max_model_len"]) == (0.25, 8192)
    assert answer["gpu_shares"] == {"finetune": 0.3, "evaluate": 0.25}


def test_a_users_serving_share_wins(logged_in_api, base_model_files, cluster):
    with_gpu(cluster, 125, shared_with_system=True)
    serving = {"gpu_memory_utilization": 0.6}
    request = {
        **pipeline_request(),
        "evaluate": {"benchmarks": ["lm_eval:gsm8k"], "serving": serving},
    }

    assert validated(logged_in_api, request)["gpu_shares"]["evaluate"] == 0.6


def test_a_share_above_the_budget_is_refused_only_if_the_user_entered_it(
    logged_in_api, base_model_files, cluster
):
    with_gpu(cluster, 24)
    logged_in_api.app.state.settings = logged_in_api.app.state.settings.model_copy(
        update={"gpu_budget": 0.5}
    )

    assert validated(logged_in_api, pipeline_request())["request"]["gpu_share"] == 0.5
    assert validate(logged_in_api, {**pipeline_request(), "gpu_share": 0.6}).status_code == 422


def test_a_teacher_beside_the_student_counts(
    logged_in_api, base_model_files, hugging_face, cluster
):
    on_the_hub(hugging_face, "big/teacher", 39525311232, LLAMA_CONFIG)
    with_gpu(cluster, 125, shared_with_system=True)
    request = pipeline_request(
        phase={
            **on_prompts(logged_in_api),
            "algorithm": "distillation",
            "teacher": f"hf:big/teacher@{COMMIT}",
        }
    )

    assert validated(logged_in_api, request)["request"]["gpu_share"] == 0.55


def test_an_rl_phases_share_leaves_room_beside_colocated_vllm(
    logged_in_api,
    hugging_face,
    submittable,  # noqa: F811
    cluster,  # noqa: F811
):
    on_the_hub(hugging_face, BASE_MODEL, 2**20, TINY_CONFIG)
    with_gpu(cluster, 125, shared_with_system=True)
    reward = {"weight": 1.0, "source": "def reward(sample, item):\n    return 1.0\n"}
    request = pipeline_request(
        phase={**on_prompts(logged_in_api), "algorithm": "grpo", "rewards": {"one": reward}}
    )
    resolved = validate(logged_in_api, request)

    # 1.6% to train, beside the 30% vLLM takes out of the same share.
    assert resolved.status_code == 200, resolved.text
    assert resolved.json()["request"]["gpu_share"] == 0.35


def test_a_submitted_pipeline_trains_on_its_stored_share_and_lists_it(
    logged_in_api, base_model_files, cluster
):
    with_gpu(cluster, 125, shared_with_system=True)

    response = submit(logged_in_api)

    pipeline_id = response.json()["id"]
    stored = logged_in_api.get(api_paths.PIPELINE.format(id=pipeline_id)).json()
    assert stored["request"]["gpu_share"] == 0.3
    assert logged_in_api.get(api_paths.PIPELINES).json()[0]["gpu_shares"] == {"finetune": 0.3}
    # Submitted again, it keeps its share rather than a new calculation.
    with_gpu(cluster, 24)
    assert validated(logged_in_api, stored["request"])["request"]["gpu_share"] == 0.3
