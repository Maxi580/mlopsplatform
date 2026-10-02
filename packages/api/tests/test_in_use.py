import pytest

from mlp_api.pipelines.hugging_face import HubModel
from mlp_core import api_paths

from .test_datasets import CHAT, jsonl, upload
from .test_models import delete, models, register
from .test_pipeline_request import BASE_MODEL, COMMIT
from .test_pipelines import finish_run, reconcile, submit


@pytest.fixture
def pipeline_id(logged_in_api, hugging_face, cluster) -> int:
    """A running Pipeline training on dataset:chat@1."""
    hugging_face.models[BASE_MODEL] = HubModel(commit=COMMIT, needs_remote_code=False)
    upload(logged_in_api, "chat", jsonl(CHAT))
    response = submit(logged_in_api)
    assert response.status_code == 202, response.text
    finish_run(cluster, "RUNNING")
    reconcile(logged_in_api)
    return response.json()["id"]


def delete_dataset(api, version=1):
    return api.delete(api_paths.DATASET_VERSION.format(name="chat", version=version))


def test_a_dataset_version_a_running_pipeline_uses_is_refused_naming_the_pipeline(
    logged_in_api, pipeline_id
):
    response = delete_dataset(logged_in_api)

    assert response.status_code == 409
    assert f"qwen-sft (#{pipeline_id})" in response.json()["detail"]


def test_another_version_of_the_dataset_can_be_deleted(logged_in_api, pipeline_id):
    upload(logged_in_api, "chat", jsonl(CHAT))

    assert delete_dataset(logged_in_api, version=2).status_code == 204


def test_a_dataset_version_can_be_deleted_once_its_pipeline_finished(
    logged_in_api, pipeline_id, cluster
):
    finish_run(cluster, "FAILED")
    reconcile(logged_in_api)

    assert delete_dataset(logged_in_api).status_code == 204


def test_a_model_version_its_running_pipeline_produced_is_refused_naming_the_pipeline(
    logged_in_api, pipeline_id, model_registry, object_store
):
    register(model_registry, object_store, "qwen-sft", 1, pipeline_id=pipeline_id)

    response = delete(logged_in_api, "qwen-sft", 1)

    assert response.status_code == 409
    assert f"qwen-sft (#{pipeline_id})" in response.json()["detail"]
    assert len(models(logged_in_api)["qwen-sft"]) == 1
    assert object_store.buckets["mlflow"] != {}


def test_a_model_version_of_a_finished_pipeline_can_be_deleted(
    logged_in_api, pipeline_id, model_registry, object_store, cluster
):
    register(model_registry, object_store, "qwen-sft", 1, pipeline_id=pipeline_id)
    finish_run(cluster, "SUCCEEDED")
    reconcile(logged_in_api)

    assert delete(logged_in_api, "qwen-sft", 1).status_code == 204
