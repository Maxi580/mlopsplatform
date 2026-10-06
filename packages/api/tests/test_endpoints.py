import json

import pytest

from mlp_api.endpoints.lifecycle import reconcile_endpoints
from mlp_api.models.mlflow import ModelVersion
from mlp_api.pipelines.hugging_face import HubModel
from mlp_core import api_paths

from .test_model_cache import cache_base_model, cached, free
from .test_models import delete

QWEN = "Qwen/Qwen3-0.6B"
COMMIT = "a" * 40
PINNED_QWEN = f"hf:{QWEN}@{COMMIT}"


@pytest.fixture
def qwen_on_the_hub(logged_in_api, hugging_face):
    hugging_face.models[QWEN] = HubModel(commit=COMMIT, needs_remote_code=False, model_type="qwen3")


def register(model_registry, object_store, name, tags, files=None) -> str:
    """Version 1 of the Registered Model, its files in the mlflow bucket; returns their prefix."""
    prefix = f"1/run-{name}/artifacts/model/"
    for file, content in (files or {"model.safetensors": b"weights"}).items():
        object_store.buckets["mlflow"][prefix + file] = content
    model_registry.versions.append(ModelVersion(name, 1, tags, prefix))
    return prefix


def register_adapter(model_registry, object_store, name, base, rank=16, **tags) -> str:
    files = {"adapter_model.safetensors": b"lora", "adapter_config.json": json.dumps({"r": rank})}
    tags = {"weights": "adapter", "base_model": base, **tags}
    return register(model_registry, object_store, name, tags, files)


def start(api, model=PINNED_QWEN, name="chat", **options):
    return api.post(api_paths.ENDPOINTS, json={"name": name, "model": model, **options})


def stop(api, name="chat"):
    return api.post(api_paths.STOP_ENDPOINT.format(name=name))


def endpoints(api) -> list[dict]:
    response = api.get(api_paths.ENDPOINTS)
    assert response.status_code == 200, response.text
    return response.json()


def pod(cluster, name="chat") -> dict:
    return cluster.endpoints[name]["deployment"]["spec"]["template"]["spec"]


def vllm_args(cluster, name="chat") -> list[str]:
    [vllm] = pod(cluster, name)["containers"]
    return vllm["args"]


def init_commands(cluster, name="chat") -> list[list[str]]:
    return [container["command"] for container in pod(cluster, name).get("initContainers", [])]


def contains(args: list[str], expected: list[str]) -> bool:
    return any(args[i : i + len(expected)] == expected for i in range(len(args)))


def reconcile(api):
    reconcile_endpoints(api.app.state.engine, api.app.state.cluster)


def test_an_endpoint_for_a_base_model_fetches_it_and_serves_it_offline_from_the_model_cache(
    logged_in_api, qwen_on_the_hub, cluster
):
    response = start(logged_in_api, f"hf:{QWEN}")

    assert response.status_code == 201, response.text
    started = response.json()
    assert (started["name"], started["status"]) == ("chat", "pending")
    assert (started["model"], started["url"]) == (PINNED_QWEN, "/endpoints/chat/v1")
    assert init_commands(cluster) == [["mlp-stage", "fetch", "200Gi", PINNED_QWEN]]
    expected = [QWEN, "--served-model-name", "chat", "--tensor-parallel-size", "1"]
    assert vllm_args(cluster)[:5] == expected
    assert contains(vllm_args(cluster), ["--revision", COMMIT])
    [vllm] = pod(cluster)["containers"]
    assert vllm["image"] == "vllm/vllm-openai:test"
    assert vllm["resources"]["limits"] == {"nvidia.com/gpu": 1}
    assert {"name": "HF_HUB_OFFLINE", "value": "1"} in vllm["env"]


def test_an_endpoint_for_full_weights_downloads_the_model_version(
    logged_in_api, cluster, model_registry, object_store
):
    tags = {"weights": "full", "tool_parser": "mistral"}
    prefix = register(model_registry, object_store, "my-model", tags)

    response = start(logged_in_api, "model:my-model")

    assert response.status_code == 201, response.text
    assert response.json()["model"] == "model:my-model@1"
    assert init_commands(cluster) == [
        ["mlp-stage", "download", f"s3://mlflow/{prefix}=/models/weights"]
    ]
    assert vllm_args(cluster)[0] == "/models/weights"
    assert "--revision" not in vllm_args(cluster)
    assert contains(vllm_args(cluster), ["--tool-call-parser", "mistral"])


def test_an_endpoint_offers_its_tokenizer_so_it_can_be_evaluated(logged_in_api, qwen_on_the_hub):
    start(logged_in_api)

    assert "--enable-tokenizer-info-endpoint" in vllm_args(logged_in_api.app.state.cluster)


def test_untouched_serving_options_are_vllms_own_defaults(logged_in_api, qwen_on_the_hub, cluster):
    start(logged_in_api, f"hf:{QWEN}")

    args = vllm_args(cluster)
    for option in (
        ["--dtype", "auto"],
        ["--gpu-memory-utilization", "0.9"],
        ["--max-num-seqs", "256"],
        ["--kv-cache-dtype", "auto"],
        ["--enable-prefix-caching"],
    ):
        assert contains(args, option)
    assert "--max-model-len" not in args


def test_an_adapter_is_served_on_its_base_model_under_the_endpoint_name(
    logged_in_api, qwen_on_the_hub, cluster, model_registry, object_store
):
    prefix = register_adapter(model_registry, object_store, "qwen-sft", PINNED_QWEN, rank=12)

    response = start(logged_in_api, "model:qwen-sft@1")

    assert response.status_code == 201, response.text
    assert init_commands(cluster) == [
        ["mlp-stage", "fetch", "200Gi", PINNED_QWEN],
        ["mlp-stage", "download", f"s3://mlflow/{prefix}=/models/adapter"],
    ]
    args = vllm_args(cluster)
    assert args[:3] == [QWEN, "--served-model-name", "chat-base"]
    assert contains(args, ["--lora-modules", "chat=/models/adapter"])
    # vLLM takes only some ranks, so the Adapter's 12 needs the next one up.
    assert contains(args, ["--max-lora-rank", "16"])
    # An uploaded Adapter has no parser of its own; its base's model_type picks it.
    assert contains(args, ["--tool-call-parser", "hermes"])


def test_an_adapter_on_full_weights_downloads_both(
    logged_in_api, cluster, model_registry, object_store
):
    base = register(model_registry, object_store, "my-model", {"weights": "full"})
    adapter = register_adapter(
        model_registry, object_store, "my-sft", "model:my-model@1", tool_parser="none"
    )

    response = start(logged_in_api, "model:my-sft@1")

    assert response.status_code == 201, response.text
    download = ["mlp-stage", "download"]
    weights, lora = f"s3://mlflow/{base}=/models/weights", f"s3://mlflow/{adapter}=/models/adapter"
    assert init_commands(cluster) == [[*download, weights, lora]]
    assert vllm_args(cluster)[0] == "/models/weights"
    assert "--enable-auto-tool-choice" not in vllm_args(cluster)


@pytest.mark.parametrize(
    ("options", "flags"),
    [
        ({}, ["--enable-prefix-caching"]),
        ({"prefix_caching": False}, ["--no-enable-prefix-caching"]),
        ({"max_model_len": 4096}, ["--max-model-len", "4096"]),
        ({"dtype": "bfloat16"}, ["--dtype", "bfloat16"]),
        ({"gpu_memory_utilization": 0.8}, ["--gpu-memory-utilization", "0.8"]),
        ({"max_num_seqs": 64}, ["--max-num-seqs", "64"]),
        ({"max_num_batched_tokens": 8192}, ["--max-num-batched-tokens", "8192"]),
        ({"async_scheduling": True}, ["--async-scheduling"]),
        ({"kv_cache_dtype": "fp8"}, ["--kv-cache-dtype", "fp8"]),
        ({"quantization": "bitsandbytes"}, ["--quantization", "bitsandbytes"]),
        (
            {"tool_parser": "llama3_json"},
            ["--enable-auto-tool-choice", "--tool-call-parser", "llama3_json"],
        ),
    ],
)
def test_serving_options_become_vllm_flags(logged_in_api, qwen_on_the_hub, cluster, options, flags):
    response = start(logged_in_api, **options)

    assert response.status_code == 201, response.text
    assert contains(vllm_args(cluster), flags)


def test_a_model_type_without_a_parser_serves_without_tool_calling(
    logged_in_api, hugging_face, cluster
):
    hugging_face.models[QWEN] = HubModel(commit=COMMIT, needs_remote_code=False, model_type="gpt2")

    start(logged_in_api)

    assert "--enable-auto-tool-choice" not in vllm_args(cluster)


def test_the_gpu_count_comes_from_platform_settings(logged_in_api, qwen_on_the_hub, cluster):
    logged_in_api.app.state.settings.gpus_per_endpoint = 2

    start(logged_in_api)

    assert contains(vllm_args(cluster), ["--tensor-parallel-size", "2"])
    [vllm] = pod(cluster)["containers"]
    assert vllm["resources"]["limits"] == {"nvidia.com/gpu": 2}


@pytest.mark.parametrize(
    "options",
    [
        {"tensor_parallel_size": 2},
        {"enforce_eager": True},
        {"tool_parser": "my_plugin"},
        {"quantization": "awq"},
        {"gpu_memory_utilization": 1.5},
        {"name": "Not_A_DNS_Label"},
    ],
)
def test_unknown_or_invalid_serving_options_are_rejected(
    logged_in_api, qwen_on_the_hub, cluster, options
):
    response = start(logged_in_api, **options)

    assert response.status_code == 422
    assert cluster.endpoints == {}


@pytest.mark.parametrize(
    ("model", "reason"),
    [
        ("hf:org/missing", "org/missing on Hugging Face is missing"),
        ("model:nothing", "no Registered Model `nothing`"),
        ("model:my-model@2", "`my-model` has no version 2"),
    ],
)
def test_a_model_that_cannot_be_found_is_rejected(
    logged_in_api, cluster, model_registry, object_store, model, reason
):
    register(model_registry, object_store, "my-model", {"weights": "full"})

    response = start(logged_in_api, model)

    assert response.status_code == 422
    assert reason in response.json()["detail"]
    assert cluster.endpoints == {}


def test_a_base_model_needing_remote_code_is_rejected(logged_in_api, hugging_face, cluster):
    hugging_face.models[QWEN] = HubModel(commit=COMMIT, needs_remote_code=True)

    response = start(logged_in_api)

    assert response.status_code == 422
    assert "remote code" in response.json()["detail"]


def test_the_route_serves_the_endpoint_url_behind_the_login(
    logged_in_api, qwen_on_the_hub, cluster
):
    start(logged_in_api)

    [route] = cluster.endpoints["chat"]["route"]["spec"]["routes"]
    assert route["match"] == "Host(`platform.test`) && PathPrefix(`/endpoints/chat/`)"
    assert [m["name"] for m in route["middlewares"]] == ["login", "endpoint-strip-prefix"]
    assert route["services"] == [{"name": "endpoint-chat", "port": 8000}]


def test_endpoints_are_listed_newest_first_and_follow_their_pod(
    logged_in_api, qwen_on_the_hub, cluster
):
    start(logged_in_api, name="first")
    start(logged_in_api, name="second")
    cluster.endpoint_states["second"] = "running"

    reconcile(logged_in_api)

    listed = endpoints(logged_in_api)
    assert [(e["name"], e["status"]) for e in listed] == [
        ("second", "running"),
        ("first", "pending"),
    ]
    assert listed[0]["owner"] == "shared"


def test_an_endpoint_whose_vllm_keeps_crashing_is_failed(logged_in_api, qwen_on_the_hub, cluster):
    start(logged_in_api)
    cluster.endpoint_states["chat"] = "failed"

    reconcile(logged_in_api)

    [endpoint] = endpoints(logged_in_api)
    assert endpoint["status"] == "failed"


def test_a_start_kubernetes_refuses_leaves_no_objects_behind(
    logged_in_api, qwen_on_the_hub, cluster
):
    def refuse_route(manifests):
        cluster.endpoints["chat"] = manifests
        raise RuntimeError("no IngressRoute")

    cluster.create_endpoint = refuse_route

    response = start(logged_in_api)
    reconcile(logged_in_api)

    assert response.status_code == 502
    assert cluster.endpoints == {}
    [endpoint] = endpoints(logged_in_api)
    assert endpoint["status"] == "stopped"


def test_an_endpoint_whose_deployment_disappeared_is_stopped(
    logged_in_api, qwen_on_the_hub, cluster
):
    start(logged_in_api)
    cluster.endpoints.clear()

    reconcile(logged_in_api)

    [endpoint] = endpoints(logged_in_api)
    assert endpoint["status"] == "stopped"


def test_stop_deletes_the_deployment_and_marks_the_endpoint_stopped(
    logged_in_api, qwen_on_the_hub, cluster
):
    start(logged_in_api)

    response = stop(logged_in_api)

    assert response.status_code == 200, response.text
    assert cluster.deleted_endpoints == ["chat"]
    assert cluster.endpoints == {}
    [endpoint] = endpoints(logged_in_api)
    assert endpoint["status"] == "stopped"


def test_stopping_an_endpoint_that_is_not_running_is_not_found(
    logged_in_api, qwen_on_the_hub, cluster
):
    start(logged_in_api)
    stop(logged_in_api)

    assert stop(logged_in_api).status_code == 404
    assert stop(logged_in_api, "unknown").status_code == 404


def delete_endpoint(api, name="chat"):
    return api.delete(api_paths.ENDPOINT.format(name=name))


def test_delete_removes_a_stopped_endpoint(logged_in_api, qwen_on_the_hub, cluster):
    start(logged_in_api)
    stop(logged_in_api)

    response = delete_endpoint(logged_in_api)

    assert response.status_code == 204, response.text
    assert endpoints(logged_in_api) == []


def test_a_running_endpoint_cannot_be_deleted(logged_in_api, qwen_on_the_hub, cluster):
    start(logged_in_api)

    response = delete_endpoint(logged_in_api)

    assert response.status_code == 409
    assert "stop it first" in response.json()["detail"]
    assert [e["status"] for e in endpoints(logged_in_api)] == ["pending"]


def test_delete_keeps_the_running_endpoint_of_the_same_name(
    logged_in_api, qwen_on_the_hub, cluster
):
    start(logged_in_api)
    stop(logged_in_api)
    start(logged_in_api)

    assert delete_endpoint(logged_in_api).status_code == 204
    assert [e["status"] for e in endpoints(logged_in_api)] == ["pending"]


def test_deleting_an_unknown_endpoint_is_not_found(logged_in_api):
    assert delete_endpoint(logged_in_api, "unknown").status_code == 404


def test_a_name_is_taken_until_its_endpoint_stops(logged_in_api, qwen_on_the_hub, cluster):
    start(logged_in_api)

    refused = start(logged_in_api)
    stop(logged_in_api)
    restarted = start(logged_in_api)

    assert refused.status_code == 409
    assert restarted.status_code == 201, restarted.text
    assert [e["status"] for e in endpoints(logged_in_api)] == ["pending", "stopped"]


def test_a_served_model_version_cannot_be_deleted_until_its_endpoint_stops(
    logged_in_api, cluster, model_registry, object_store
):
    register(model_registry, object_store, "my-model", {"weights": "full"})
    start(logged_in_api, "model:my-model@1")

    refused = delete(logged_in_api, "my-model", 1)
    stop(logged_in_api)

    assert refused.status_code == 409
    assert "Endpoint chat" in refused.json()["detail"]
    assert delete(logged_in_api, "my-model", 1).status_code == 204


def test_the_base_model_of_a_served_adapter_stays_in_the_model_cache(
    logged_in_api, qwen_on_the_hub, cluster, model_registry, object_store, model_cache
):
    cache_base_model(model_cache, PINNED_QWEN, 100, days_since_use=1)
    register_adapter(model_registry, object_store, "qwen-sft", PINNED_QWEN)
    start(logged_in_api, "model:qwen-sft@1")

    response = free(logged_in_api, PINNED_QWEN)

    assert response.status_code == 409
    assert "Endpoint chat" in response.json()["detail"]
    assert PINNED_QWEN in cached(logged_in_api)


def test_endpoints_require_login(api, cluster):
    assert api.get(api_paths.ENDPOINTS).status_code == 401
    assert start(api).status_code == 401
    assert stop(api).status_code == 401
    assert delete_endpoint(api).status_code == 401
