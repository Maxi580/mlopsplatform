import json

import pytest

from mlp_api.pipelines.hugging_face import HubModel
from mlp_core import api_paths, config

from .test_models import models, register

INDEX = {"weight_map": {"embed": "model-1-of-2.safetensors", "head": "model-2-of-2.safetensors"}}
FULL_WEIGHTS = {
    "config.json": json.dumps({"model_type": "qwen2"}).encode(),
    "tokenizer_config.json": b"{}",
    "tokenizer.json": b'{"model": {}}',
    "model.safetensors.index.json": json.dumps(INDEX).encode(),
    "model-1-of-2.safetensors": b"first shard",
    "model-2-of-2.safetensors": b"second shard",
}
ADAPTER_CONFIG = {"r": 16, "bias": "none", "use_dora": False, "modules_to_save": None}
ADAPTER = {
    "adapter_config.json": json.dumps(ADAPTER_CONFIG).encode(),
    "adapter_model.safetensors": b"lora weights",
}
QWEN = "Qwen/Qwen3-0.6B"


@pytest.fixture
def small_parts(monkeypatch):
    # Splits the test files into several parts, as a large model would be.
    monkeypatch.setattr(config, "MODEL_UPLOAD_PART_SIZE", 4)


def start(api, files: dict[str, bytes], name="my-model", **fields):
    listing = [{"path": path, "size_bytes": len(content)} for path, content in files.items()]
    return api.post(api_paths.MODEL_UPLOADS, json={"name": name, "files": listing, **fields})


def send_parts(object_store, started: dict, files: dict[str, bytes]) -> None:
    """What a client does between start and complete: PUT each part to its URL."""
    size = started["part_size_bytes"]
    for file in started["files"]:
        content = files[file["path"]]
        for index, url in enumerate(file["part_urls"]):
            object_store.receive_part(url, content[index * size : (index + 1) * size])


def upload(api, object_store, files: dict[str, bytes], **fields):
    """The answer to completing the upload, or to starting it when that was refused."""
    started = start(api, files, **fields)
    if started.status_code != 201:
        return started
    send_parts(object_store, started.json(), files)
    return api.post(api_paths.MODEL_UPLOAD_COMPLETE.format(id=started.json()["id"]))


def with_file(files: dict, path: str, content) -> dict:
    content = content if isinstance(content, bytes) else json.dumps(content).encode()
    return {**files, path: content}


def without_file(files: dict, path: str) -> dict:
    return {key: value for key, value in files.items() if key != path}


def rejection(response) -> str:
    assert response.status_code == 422, response.text
    return response.json()["detail"]


def test_full_weights_upload_and_register_as_uploaded(
    logged_in_api, object_store, model_registry, small_parts
):
    response = upload(logged_in_api, object_store, FULL_WEIGHTS)

    assert response.status_code == 201, response.text
    assert response.json() == {"name": "my-model", "version": 1}
    [version] = models(logged_in_api)["my-model"]
    assert version["tags"] == {
        "weights": "full",
        "source": "uploaded",
        "owner": config.OWNER,
        "tool_parser": "hermes",
    }
    assert version["size_bytes"] == sum(len(content) for content in FULL_WEIGHTS.values())


def test_another_upload_under_the_same_name_is_the_next_version(
    logged_in_api, object_store, model_registry
):
    upload(logged_in_api, object_store, FULL_WEIGHTS)

    assert upload(logged_in_api, object_store, FULL_WEIGHTS).json()["version"] == 2


@pytest.mark.parametrize(
    ("model_config", "fields", "tool_parser"),
    [
        ({"model_type": "phi3"}, {}, "none"),
        ({"model_type": "phi3"}, {"tool_parser": "phi4_mini_json"}, "phi4_mini_json"),
    ],
)
def test_an_unknown_model_type_serves_without_tool_calling_unless_a_parser_is_named(
    logged_in_api, object_store, model_registry, model_config, fields, tool_parser
):
    files = with_file(FULL_WEIGHTS, "config.json", model_config)

    assert upload(logged_in_api, object_store, files, **fields).status_code == 201
    assert models(logged_in_api)["my-model"][0]["tags"]["tool_parser"] == tool_parser


@pytest.mark.parametrize(
    ("files", "fields", "reason"),
    [
        (without_file(FULL_WEIGHTS, "config.json"), {}, "config.json"),
        (without_file(FULL_WEIGHTS, "tokenizer.json"), {}, "tokenizer"),
        (with_file(FULL_WEIGHTS, "pytorch_model.bin", b"pickle"), {}, "pytorch_model.bin"),
        (without_file(FULL_WEIGHTS, "model.safetensors.index.json"), {}, "index"),
        (FULL_WEIGHTS, {"base": f"hf:{QWEN}"}, "only for Adapters"),
        (ADAPTER, {}, "base"),
        (with_file(ADAPTER, "adapter_model.bin", b"pickle"), {"base": f"hf:{QWEN}"}, ".bin"),
        (with_file(FULL_WEIGHTS, "../escape.json", b"{}"), {}, "../escape.json"),
    ],
)
def test_files_that_cannot_make_a_model_version_are_refused_before_uploading(
    logged_in_api, object_store, model_registry, hugging_face, files, fields, reason
):
    hugging_face.models[QWEN] = HubModel(commit="abc", needs_remote_code=False)

    response = start(logged_in_api, files, **fields)

    assert reason in rejection(response)
    assert object_store.multipart_uploads == {}


@pytest.mark.parametrize(
    ("files", "reason"),
    [
        (
            with_file(FULL_WEIGHTS, "config.json", {"model_type": "qwen2", "auto_map": {}}),
            "auto_map",
        ),
        (with_file(FULL_WEIGHTS, "tokenizer_config.json", {"auto_map": {}}), "auto_map"),
        (without_file(FULL_WEIGHTS, "model-2-of-2.safetensors"), "model-2-of-2.safetensors"),
        (with_file(FULL_WEIGHTS, "config.json", b"{not json"), "config.json"),
    ],
)
def test_full_weights_failing_the_checks_are_rejected_and_deleted(
    logged_in_api, object_store, model_registry, files, reason
):
    assert reason in rejection(upload(logged_in_api, object_store, files))
    assert model_registry.versions == []
    assert object_store.buckets["mlflow"] == {}


@pytest.mark.parametrize(
    ("adapter_config", "reason"),
    [
        ({"use_dora": True}, "DoRA"),
        ({"modules_to_save": ["lm_head"]}, "modules_to_save"),
        ({"bias": "all"}, "bias"),
        ({"r": config.MAX_LORA_RANK + 1}, "rank"),
        ({"auto_map": {}}, "auto_map"),
    ],
)
def test_adapters_vllm_cannot_serve_are_rejected(
    logged_in_api, object_store, model_registry, hugging_face, adapter_config, reason
):
    hugging_face.models[QWEN] = HubModel(commit="abc", needs_remote_code=False)
    files = with_file(ADAPTER, "adapter_config.json", {**ADAPTER_CONFIG, **adapter_config})

    response = upload(logged_in_api, object_store, files, base=f"hf:{QWEN}")

    assert reason in rejection(response)
    assert object_store.buckets["mlflow"] == {}


def test_an_adapter_registers_with_its_base_model_pinned_to_a_commit(
    logged_in_api, object_store, model_registry, hugging_face
):
    hugging_face.models[QWEN] = HubModel(commit="abc", needs_remote_code=False)

    response = upload(logged_in_api, object_store, ADAPTER, base=f"hf:{QWEN}")

    assert response.status_code == 201, response.text
    tags = models(logged_in_api)["my-model"][0]["tags"]
    assert (tags["weights"], tags["base_model"]) == ("adapter", f"hf:{QWEN}@abc")


def test_an_adapter_on_a_full_weight_model_version_registers_with_it_pinned(
    logged_in_api, object_store, model_registry
):
    upload(logged_in_api, object_store, FULL_WEIGHTS, name="my-base")

    response = upload(logged_in_api, object_store, ADAPTER, base="model:my-base")

    assert response.status_code == 201, response.text
    assert models(logged_in_api)["my-model"][0]["tags"]["base_model"] == "model:my-base@1"


@pytest.mark.parametrize(
    ("base", "reason"),
    [
        ("hf:nobody/missing", "Hugging Face"),
        ("model:missing", "no Registered Model"),
        ("model:qwen-sft@1", "Adapter"),
        ("s3://bucket/model", "base"),
    ],
)
def test_an_adapter_base_that_is_missing_or_not_full_weights_is_rejected(
    logged_in_api, object_store, model_registry, base, reason
):
    register(model_registry, object_store, "qwen-sft", 1)

    assert reason in str(start(logged_in_api, ADAPTER, base=base).json()["detail"])


def test_an_upload_missing_a_part_is_rejected_and_deleted(
    logged_in_api, object_store, model_registry, small_parts
):
    started = start(logged_in_api, FULL_WEIGHTS).json()
    send_parts(object_store, started, FULL_WEIGHTS)
    shard = next(f for f in started["files"] if f["path"] == "model-1-of-2.safetensors")
    object_store.parts.pop(shard["part_urls"][-1])

    response = logged_in_api.post(api_paths.MODEL_UPLOAD_COMPLETE.format(id=started["id"]))

    assert "model-1-of-2.safetensors did not arrive in full" in rejection(response)
    assert model_registry.versions == []
    assert object_store.buckets["mlflow"] == {}


def test_completing_an_unknown_upload_is_not_found(logged_in_api):
    response = logged_in_api.post(api_paths.MODEL_UPLOAD_COMPLETE.format(id="nope"))

    assert response.status_code == 404


def test_a_model_version_downloads_as_one_url_per_file(logged_in_api, object_store, model_registry):
    upload(logged_in_api, object_store, FULL_WEIGHTS)

    response = logged_in_api.get(api_paths.MODEL_VERSION_FILES.format(name="my-model", version=1))

    assert response.status_code == 200, response.text
    files = {file["path"]: file for file in response.json()["files"]}
    assert set(files) == set(FULL_WEIGHTS)
    url = files["config.json"]["url"]
    assert url.startswith("https://objects.test/mlflow/") and url.endswith(
        "/config.json?signature=x"
    )
    assert files["config.json"]["size_bytes"] == len(FULL_WEIGHTS["config.json"])


def test_files_of_an_unknown_model_version_are_not_found(logged_in_api, model_registry):
    response = logged_in_api.get(api_paths.MODEL_VERSION_FILES.format(name="my-model", version=1))

    assert response.status_code == 404


def test_uploads_and_downloads_require_login(api, model_registry):
    assert start(api, FULL_WEIGHTS).status_code == 401
    assert api.post(api_paths.MODEL_UPLOAD_COMPLETE.format(id="x")).status_code == 401
    files = api_paths.MODEL_VERSION_FILES.format(name="my-model", version=1)
    assert api.get(files).status_code == 401
