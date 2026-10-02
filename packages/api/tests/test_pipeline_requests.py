import pytest

from mlp_api.hugging_face import HubModel
from mlp_core import api_paths, config
from mlp_core.trainers import trainer_config_schema

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
COMMIT = "7ae557604adf67be50417f59c2c2f167def9a775"
HF_TOKEN = "hf_" + "a1B2c3D4" * 5


def pipeline_request(phase=None, **finetune):
    return {
        "name": "qwen-sft",
        "finetune": {
            "base_model": f"hf:{BASE_MODEL}",
            "phases": [{"algorithm": "sft", "dataset": "dataset:chat", **(phase or {})}],
            **finetune,
        },
    }


@pytest.fixture
def validate(logged_in_api, hugging_face):
    hugging_face.models[BASE_MODEL] = HubModel(commit=COMMIT, needs_remote_code=False)

    def run(request, secrets=None):
        body = {"request": request, "secrets": secrets or {}}
        return logged_in_api.post(api_paths.VALIDATE_PIPELINE, json=body)

    return run


def rejection(response) -> str:
    assert response.status_code == 422, response.text
    return " | ".join(f"{error['loc']}: {error['msg']}" for error in response.json()["detail"])


def test_schema_publishes_the_pipeline_request_with_trl_settings(logged_in_api):
    response = logged_in_api.get(api_paths.SCHEMA)

    assert response.status_code == 200
    schema = response.json()
    assert set(schema["properties"]) >= {"name", "finetune"}
    settings = schema["$defs"]["SftPhase"]["properties"]["settings"]["properties"]
    assert "learning_rate" in settings
    assert "trust_remote_code" not in settings


def test_a_valid_request_comes_back_resolved(validate):
    response = validate(
        pipeline_request(phase={"settings": {"learning_rate": 1e-4}, "lora": {"r": 16}})
    )

    assert response.status_code == 200, response.text
    finetune = response.json()["request"]["finetune"]
    assert finetune["base_model"] == f"hf:{BASE_MODEL}@{COMMIT}"
    assert finetune["backend"] == "hf"
    assert finetune["phases"][0]["method"] == "lora"


def test_the_hf_token_secret_is_used_to_look_up_the_base_model(validate, hugging_face):
    validate(pipeline_request(), secrets={"hf_token": HF_TOKEN})

    assert hugging_face.lookups == [(BASE_MODEL, "main", HF_TOKEN)]


@pytest.mark.parametrize(
    "request_",
    [
        {**pipeline_request(), "gpus": 2},
        pipeline_request(phase={"epochs": 3}),
        pipeline_request(tensor_parallel_size=2),
    ],
)
def test_unknown_fields_are_rejected(validate, request_):
    assert "Extra inputs are not permitted" in rejection(validate(request_))


def test_settings_unknown_to_the_trl_config_are_rejected(validate):
    message = rejection(validate(pipeline_request(phase={"settings": {"lerning_rate": 1e-4}})))

    assert "lerning_rate" in message
    assert "SFTConfig" in message


def test_settings_of_the_wrong_type_are_rejected(validate):
    message = rejection(validate(pipeline_request(phase={"settings": {"num_train_epochs": "x"}})))

    assert "num_train_epochs" in message


def test_trust_remote_code_is_rejected_wherever_it_appears(validate):
    settings = {"model_init_kwargs": {"trust_remote_code": True}}

    message = rejection(validate(pipeline_request(phase={"settings": settings})))

    assert "trust_remote_code" in message


def test_a_base_model_unreachable_on_hugging_face_is_rejected(validate):
    response = validate(pipeline_request(base_model="hf:nobody/missing-model"))

    assert "nobody/missing-model" in rejection(response)


def test_a_base_model_that_needs_remote_code_is_rejected(validate, hugging_face):
    hugging_face.models["org/custom"] = HubModel(commit=COMMIT, needs_remote_code=True)

    assert "remote code" in rejection(validate(pipeline_request(base_model="hf:org/custom")))


def test_a_stage_needing_more_gpus_than_the_platform_has_is_rejected(validate, logged_in_api):
    logged_in_api.app.state.settings.gpus_per_stage = 2

    assert "GPU" in rejection(validate(pipeline_request()))


def test_a_secret_value_inside_the_request_is_rejected(validate):
    response = validate(
        pipeline_request(phase={"settings": {"run_name": f"run-{HF_TOKEN}"}}),
        secrets={"hf_token": HF_TOKEN},
    )

    assert "Secret" in rejection(response)
    assert HF_TOKEN not in response.text


def test_a_hugging_face_token_inside_the_request_is_rejected_without_secrets(validate):
    response = validate(pipeline_request(phase={"settings": {"run_name": HF_TOKEN}}))

    assert "Secret" in rejection(response)


def test_every_error_is_reported_with_its_path(validate):
    phase = {"settings": {"lerning_rate": 1}, "lora": {"rank": 8}}

    detail = validate(pipeline_request(phase=phase)).json()["detail"]

    assert [error["loc"][-1] for error in detail] == ["lerning_rate", "rank"]


def test_validation_requires_login(api, hugging_face):
    response = api.post(api_paths.VALIDATE_PIPELINE, json={"request": pipeline_request()})

    assert response.status_code == 401


@pytest.fixture
def no_generated_schemas(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRAINER_CONFIGS_DIRECTORY", tmp_path)
    trainer_config_schema.cache_clear()
    yield
    trainer_config_schema.cache_clear()


def test_settings_without_a_generated_schema_are_accepted_unchecked(validate, no_generated_schemas):
    phase = {"settings": {"anything": "goes"}, "lora": {"new_peft_option": 1}}

    assert validate(pipeline_request(phase=phase)).status_code == 200
