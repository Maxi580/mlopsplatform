import json

import pytest

from mlp_core import api_paths

CHAT = {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}


def jsonl(*rows) -> bytes:
    return "".join(json.dumps(row) + "\n" for row in rows).encode()


def upload(api, name, content):
    return api.post(api_paths.DATASET_VERSIONS.format(name=name), content=content)


def datasets(api) -> dict:
    response = api.get(api_paths.DATASETS)
    assert response.status_code == 200
    return {dataset["name"]: dataset["versions"] for dataset in response.json()}


def test_each_upload_creates_the_next_version(logged_in_api):
    first = upload(logged_in_api, "chat", jsonl(CHAT))
    second = upload(logged_in_api, "chat", jsonl(CHAT, CHAT))

    assert first.status_code == 201, first.text
    assert (first.json()["version"], second.json()["version"]) == (1, 2)


def test_the_list_shows_versions_with_sizes_and_row_formats(logged_in_api):
    content = jsonl(CHAT)
    upload(logged_in_api, "chat", content)
    pairs = jsonl({"prompt": "2+2?", "completion": "4"})
    upload(logged_in_api, "pairs", pairs)

    assert datasets(logged_in_api) == {
        "chat": [{"version": 1, "size_bytes": len(content), "row_format": "messages"}],
        "pairs": [{"version": 1, "size_bytes": len(pairs), "row_format": "prompt_completion"}],
    }


def test_the_file_lands_in_the_platform_bucket(logged_in_api, object_store):
    upload(logged_in_api, "chat", jsonl(CHAT))

    assert object_store.objects == {"datasets/chat/1/data.jsonl": jsonl(CHAT)}


@pytest.mark.parametrize(
    "row",
    [
        {"text": "Plain text for continued pretraining"},
        {"prompt": "Say hi"},
        {"prompt": [{"role": "user", "content": "Hi"}], "completion": "Hello"},
        {"prompt": "Hi", "chosen": "Hello", "rejected": "Go away"},
        {"chosen": [{"role": "user", "content": "Hi"}], "rejected": "x"},
        {"prompt": "Hi", "completion": "Hello", "label": True},
        {"prompt": "1+1?", "completions": ["1+1", "=2"], "labels": [True, True]},
        {**CHAT, "tools": [{"type": "function", "function": {"name": "f"}}]},
    ],
)
def test_every_supported_row_format_is_accepted(logged_in_api, row):
    assert upload(logged_in_api, "data", jsonl(row)).status_code == 201


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (jsonl(CHAT) + b"{not json\n", "line 2: is not JSON"),
        (jsonl(CHAT, ["a list"]), "line 2: is not a JSON object"),
        (jsonl({"question": "Hi"}), "line 1: has the fields of no row format"),
        (jsonl({"messages": "Hi"}), "line 1: `messages` must be messages"),
        (jsonl({"messages": [{"content": "no role"}]}), "line 1: `messages` must be messages"),
        (
            jsonl({"prompt": "Hi", "completion": "x", "label": "yes"}),
            "`label` must be true or false",
        ),
        (jsonl(CHAT, {"text": "x"}), "line 2: is text rows, but line 1 is messages rows"),
        (jsonl(CHAT) + b"\n", "line 2: is not JSON"),
        (b"", "has no rows"),
    ],
)
def test_malformed_files_are_rejected_with_the_offending_row(logged_in_api, content, message):
    response = upload(logged_in_api, "chat", content)

    assert response.status_code == 422
    assert message in response.json()["detail"]
    assert datasets(logged_in_api) == {}


def test_a_rejected_upload_takes_no_version_number(logged_in_api):
    upload(logged_in_api, "chat", b"{not json\n")

    assert upload(logged_in_api, "chat", jsonl(CHAT)).json()["version"] == 1


@pytest.mark.parametrize("name", ["..", ".hidden", "with space", "a/b"])
def test_dataset_names_are_checked(logged_in_api, name):
    assert upload(logged_in_api, name, jsonl(CHAT)).status_code in (404, 422)


def test_download_returns_a_presigned_url_for_the_version(logged_in_api):
    upload(logged_in_api, "chat", jsonl(CHAT))

    response = logged_in_api.get(api_paths.DATASET_DOWNLOAD.format(name="chat", version=1))

    assert response.status_code == 200
    assert response.json() == {
        "url": "https://objects.test/platform/datasets/chat/1/data.jsonl?signature=x"
    }


def test_downloading_an_unknown_version_is_not_found(logged_in_api):
    upload(logged_in_api, "chat", jsonl(CHAT))

    response = logged_in_api.get(api_paths.DATASET_DOWNLOAD.format(name="chat", version=2))

    assert response.status_code == 404


def test_deleting_a_version_removes_it_from_the_list_and_the_object_store(
    logged_in_api, object_store
):
    upload(logged_in_api, "chat", jsonl(CHAT))
    upload(logged_in_api, "chat", jsonl(CHAT))

    response = logged_in_api.delete(api_paths.DATASET_VERSION.format(name="chat", version=1))

    assert response.status_code == 204
    assert [v["version"] for v in datasets(logged_in_api)["chat"]] == [2]
    assert list(object_store.objects) == ["datasets/chat/2/data.jsonl"]


def test_a_deleted_version_number_is_never_handed_out_again(logged_in_api):
    upload(logged_in_api, "chat", jsonl(CHAT))
    logged_in_api.delete(api_paths.DATASET_VERSION.format(name="chat", version=1))

    assert upload(logged_in_api, "chat", jsonl(CHAT)).json()["version"] == 2
    assert datasets(logged_in_api)["chat"][0]["version"] == 2


def test_deleting_an_unknown_version_is_not_found(logged_in_api):
    response = logged_in_api.delete(api_paths.DATASET_VERSION.format(name="chat", version=1))

    assert response.status_code == 404


def test_datasets_require_login(api, object_store):
    assert upload(api, "chat", jsonl(CHAT)).status_code == 401
    assert api.get(api_paths.DATASETS).status_code == 401
