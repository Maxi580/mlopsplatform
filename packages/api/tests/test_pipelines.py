from mlp_api.pipelines.hugging_face import HubModel
from mlp_core import api_paths

from .test_datasets import CHAT, jsonl, upload
from .test_pipeline_request import BASE_MODEL, COMMIT, pipeline_request, without


def validate(api, request):
    return api.post(api_paths.VALIDATE_PIPELINE, json={"request": request})


def test_schema_publishes_the_pipeline_request_and_its_basic_values(logged_in_api):
    response = logged_in_api.get(api_paths.SCHEMA)

    assert response.status_code == 200
    schema = response.json()
    assert set(schema["properties"]) >= {"name", "finetune"}
    assert "learning_rate" in schema["$defs"]["SftSettings"]["required"]


def test_validate_returns_the_resolved_request(logged_in_api, hugging_face):
    hugging_face.models[BASE_MODEL] = HubModel(commit=COMMIT, needs_remote_code=False)
    upload(logged_in_api, "chat", jsonl(CHAT))

    response = validate(logged_in_api, pipeline_request())

    assert response.status_code == 200, response.text
    finetune = response.json()["request"]["finetune"]
    assert finetune["base_model"] == f"hf:{BASE_MODEL}@{COMMIT}"
    assert finetune["phases"][0]["dataset"] == "dataset:chat@1"


def test_validate_rejects_with_paths_inside_the_pipeline_request(logged_in_api):
    response = validate(logged_in_api, without(pipeline_request(), "finetune", "base_model"))

    assert response.status_code == 422
    assert response.json()["detail"] == [
        {"loc": ["finetune", "base_model"], "msg": "Field required"}
    ]


def test_validation_requires_login(api, hugging_face):
    assert validate(api, pipeline_request()).status_code == 401
