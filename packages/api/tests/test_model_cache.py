import os
from datetime import UTC, datetime, timedelta

import pytest

from mlp_api.model_cache.janitor import evict_least_recently_used
from mlp_api.pipelines.hugging_face import HubModel
from mlp_core import api_paths, config
from mlp_core.pipeline_request.references import benchmark_directory

from .test_datasets import CHAT, jsonl, upload
from .test_pipeline_request import BASE_MODEL, COMMIT, pipeline_request
from .test_pipelines import finish_run, reconcile, submit

USED_BASE_MODEL = f"hf:{BASE_MODEL}@{COMMIT}"
OLD = "hf:org/old@" + "1" * 40
OLDER = "hf:org/older@" + "2" * 40
NEW = "hf:org/new@" + "3" * 40
GSM8K = "lm_eval:gsm8k"
IFEVAL = "lm_eval:ifeval"


def cache_base_model(model_cache, reference, size_bytes, days_since_use):
    """Writes the revision into the Hugging Face cache layout, last read the given days ago."""
    repo, commit = reference.removeprefix("hf:").split("@")
    snapshot = model_cache / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / commit
    snapshot.mkdir(parents=True)
    last_read(snapshot / "model.safetensors", size_bytes, days_since_use)


def cache_benchmark(model_cache, benchmark, size_bytes, days_since_use):
    """Writes the benchmark's datasets as fetch leaves them, last read the given days ago."""
    directory = benchmark_directory(model_cache, benchmark)
    (directory / "datasets").mkdir(parents=True)
    last_read(directory / config.BENCHMARK_FETCHED_MARKER, 0, days_since_use)
    last_read(directory / "datasets" / "test.arrow", size_bytes, days_since_use)


def last_read(path, size_bytes, days_since_use):
    path.write_bytes(b"x" * size_bytes)
    used = (datetime.now(UTC) - timedelta(days=days_since_use)).timestamp()
    os.utime(path, (used, used))


def cached(api) -> dict[str, int]:
    response = api.get(api_paths.MODEL_CACHE)
    assert response.status_code == 200, response.text
    return {entry["reference"]: entry["size_bytes"] for entry in response.json()["entries"]}


def free(api, reference):
    return api.delete(api_paths.MODEL_CACHE, params={"reference": reference})


@pytest.fixture
def capacity(logged_in_api):
    """A 1000-byte Model Cache, so with the 0.9 high-water mark eviction starts above 900 bytes."""
    logged_in_api.app.state.settings.model_cache_size = "1000"


@pytest.fixture
def running_pipeline(logged_in_api, hugging_face, cluster) -> int:
    """A running Pipeline training on USED_BASE_MODEL."""
    hugging_face.models[BASE_MODEL] = HubModel(commit=COMMIT, needs_remote_code=False)
    upload(logged_in_api, "chat", jsonl(CHAT))
    response = submit(logged_in_api)
    assert response.status_code == 202, response.text
    finish_run(cluster, "RUNNING")
    reconcile(logged_in_api)
    return response.json()["id"]


def evict(api):
    state = api.app.state
    evict_least_recently_used(state.engine, state.model_cache, state.settings)


def test_cached_base_models_are_listed_with_sizes_and_last_use(
    logged_in_api, model_cache, capacity
):
    cache_base_model(model_cache, OLD, 300, days_since_use=2)

    response = logged_in_api.get(api_paths.MODEL_CACHE)

    assert response.status_code == 200, response.text
    [entry] = response.json()["entries"]
    assert (entry["kind"], entry["reference"], entry["size_bytes"]) == ("base_model", OLD, 300)
    last_used = datetime.fromisoformat(entry["last_used"])
    assert abs(datetime.now(UTC) - timedelta(days=2) - last_used) < timedelta(minutes=1)
    assert response.json()["capacity_bytes"] == 1000


def test_an_empty_model_cache_lists_nothing(logged_in_api, model_cache):
    assert cached(logged_in_api) == {}


def test_freeing_an_unused_base_model_deletes_it(logged_in_api, model_cache):
    cache_base_model(model_cache, OLD, 300, days_since_use=2)
    cache_base_model(model_cache, NEW, 100, days_since_use=0)

    assert free(logged_in_api, OLD).status_code == 204

    assert cached(logged_in_api) == {NEW: 100}
    assert not (model_cache / "hub" / "models--org--old").exists()


def test_freeing_a_base_model_a_running_pipeline_uses_is_refused_naming_it(
    logged_in_api, model_cache, running_pipeline
):
    cache_base_model(model_cache, USED_BASE_MODEL, 300, days_since_use=2)

    response = free(logged_in_api, USED_BASE_MODEL)

    assert response.status_code == 409
    assert f"qwen-sft (#{running_pipeline})" in response.json()["detail"]
    assert cached(logged_in_api) == {USED_BASE_MODEL: 300}


def test_freeing_an_uncached_base_model_is_not_found(logged_in_api, model_cache):
    assert free(logged_in_api, OLD).status_code == 404


def test_passing_the_high_water_mark_evicts_least_recently_used_until_below_it(
    logged_in_api, model_cache, capacity
):
    cache_base_model(model_cache, OLDER, 300, days_since_use=3)
    cache_base_model(model_cache, OLD, 300, days_since_use=2)
    cache_base_model(model_cache, NEW, 400, days_since_use=0)

    evict(logged_in_api)

    assert cached(logged_in_api) == {OLD: 300, NEW: 400}


def test_below_the_high_water_mark_nothing_is_evicted(logged_in_api, model_cache, capacity):
    cache_base_model(model_cache, OLD, 400, days_since_use=30)
    cache_base_model(model_cache, NEW, 500, days_since_use=0)

    evict(logged_in_api)

    assert cached(logged_in_api) == {OLD: 400, NEW: 500}


def test_base_models_in_use_are_never_evicted(
    logged_in_api, model_cache, capacity, running_pipeline
):
    cache_base_model(model_cache, USED_BASE_MODEL, 600, days_since_use=3)
    cache_base_model(model_cache, OLD, 300, days_since_use=2)
    cache_base_model(model_cache, NEW, 100, days_since_use=0)

    evict(logged_in_api)

    assert cached(logged_in_api) == {USED_BASE_MODEL: 600, NEW: 100}


def test_a_base_model_is_evictable_once_its_pipeline_finished(
    logged_in_api, model_cache, capacity, running_pipeline, cluster
):
    cache_base_model(model_cache, USED_BASE_MODEL, 1000, days_since_use=3)
    finish_run(cluster, "SUCCEEDED")
    reconcile(logged_in_api)

    evict(logged_in_api)

    assert cached(logged_in_api) == {}


def test_submitting_makes_room_for_an_uncached_base_model(
    logged_in_api, model_cache, capacity, hugging_face
):
    cache_base_model(model_cache, OLDER, 300, days_since_use=3)
    cache_base_model(model_cache, OLD, 300, days_since_use=2)
    hugging_face.models[BASE_MODEL] = HubModel(commit=COMMIT, needs_remote_code=False)
    hugging_face.sizes[BASE_MODEL] = 400
    upload(logged_in_api, "chat", jsonl(CHAT))

    assert submit(logged_in_api).status_code == 202

    assert cached(logged_in_api) == {OLD: 300}


def test_submitting_a_cached_base_model_evicts_nothing(
    logged_in_api, model_cache, capacity, hugging_face
):
    cache_base_model(model_cache, USED_BASE_MODEL, 400, days_since_use=3)
    cache_base_model(model_cache, OLD, 400, days_since_use=2)
    hugging_face.models[BASE_MODEL] = HubModel(commit=COMMIT, needs_remote_code=False)
    hugging_face.sizes[BASE_MODEL] = 400
    upload(logged_in_api, "chat", jsonl(CHAT))

    assert submit(logged_in_api).status_code == 202

    assert cached(logged_in_api) == {USED_BASE_MODEL: 400, OLD: 400}


def test_cached_benchmarks_are_listed_beside_base_models(logged_in_api, model_cache):
    cache_base_model(model_cache, OLD, 300, days_since_use=2)
    cache_benchmark(model_cache, GSM8K, 200, days_since_use=1)

    entries = logged_in_api.get(api_paths.MODEL_CACHE).json()["entries"]

    assert [(e["kind"], e["reference"], e["size_bytes"]) for e in entries] == [
        ("base_model", OLD, 300),
        ("benchmark", GSM8K, 200),
    ]


def test_freeing_a_benchmark_deletes_its_datasets(logged_in_api, model_cache):
    cache_benchmark(model_cache, GSM8K, 200, days_since_use=1)
    cache_benchmark(model_cache, IFEVAL, 100, days_since_use=1)

    assert free(logged_in_api, GSM8K).status_code == 204

    assert cached(logged_in_api) == {IFEVAL: 100}
    assert not benchmark_directory(model_cache, GSM8K).exists()


def test_benchmarks_are_evicted_least_recently_used_first_like_base_models(
    logged_in_api, model_cache, capacity
):
    cache_benchmark(model_cache, GSM8K, 300, days_since_use=3)
    cache_base_model(model_cache, OLD, 300, days_since_use=2)
    cache_benchmark(model_cache, IFEVAL, 400, days_since_use=0)

    evict(logged_in_api)

    assert cached(logged_in_api) == {OLD: 300, IFEVAL: 400}


def test_a_benchmark_a_running_pipeline_evaluates_is_never_evicted(
    logged_in_api, model_cache, capacity, hugging_face, cluster
):
    hugging_face.models[BASE_MODEL] = HubModel(commit=COMMIT, needs_remote_code=False)
    upload(logged_in_api, "chat", jsonl(CHAT))
    request = {**pipeline_request(), "evaluate": {"benchmarks": [GSM8K]}}
    assert submit(logged_in_api, request).status_code == 202
    cache_benchmark(model_cache, GSM8K, 1000, days_since_use=3)

    evict(logged_in_api)

    assert cached(logged_in_api) == {GSM8K: 1000}
    assert free(logged_in_api, GSM8K).status_code == 409


def test_the_model_cache_requires_login(api):
    assert api.get(api_paths.MODEL_CACHE).status_code == 401


def validate(api):
    return api.post(api_paths.VALIDATE_PIPELINE, json={"request": pipeline_request()})


@pytest.fixture
def hub_and_dataset(logged_in_api, hugging_face):
    hugging_face.models[BASE_MODEL] = HubModel(commit=COMMIT, needs_remote_code=False)
    upload(logged_in_api, "chat", jsonl(CHAT))


def test_validate_lists_an_uncached_base_model_as_still_to_download(
    logged_in_api, model_cache, hugging_face, hub_and_dataset
):
    hugging_face.sizes[BASE_MODEL] = 400

    answer = validate(logged_in_api).json()

    assert answer["downloads"] == [
        {"kind": "base_model", "ref": USED_BASE_MODEL, "bytes": 400, "cached": False}
    ]
    assert (answer["download_bytes"], answer["cached_bytes"]) == (400, 0)


def test_a_cached_base_model_is_marked_cached_and_left_out_of_the_total(
    logged_in_api, model_cache, hugging_face, hub_and_dataset
):
    cache_base_model(model_cache, USED_BASE_MODEL, 300, days_since_use=1)

    answer = validate(logged_in_api).json()

    assert answer["downloads"] == [
        {"kind": "base_model", "ref": USED_BASE_MODEL, "bytes": 300, "cached": True}
    ]
    assert (answer["download_bytes"], answer["cached_bytes"]) == (0, 300)


def test_submit_answers_with_the_downloads_too(
    logged_in_api, model_cache, hugging_face, hub_and_dataset
):
    hugging_face.sizes[BASE_MODEL] = 400

    answer = submit(logged_in_api).json()

    assert [download["ref"] for download in answer["downloads"]] == [USED_BASE_MODEL]
    assert answer["download_bytes"] == 400
