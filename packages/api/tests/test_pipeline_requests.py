import pytest

from mlp_api.hugging_face import HubModel
from mlp_core import api_paths, config
from mlp_core.pipeline_request.standard import STANDARD_PIPELINE_REQUEST

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
COMMIT = "7ae557604adf67be50417f59c2c2f167def9a775"
HF_TOKEN = "hf_" + "a1B2c3D4" * 5


def pipeline_request(settings=None, lora=None, phase=None, **finetune):
    """The standard request with the given values added or replaced."""
    request = STANDARD_PIPELINE_REQUEST.model_dump(mode="json")
    first_phase = request["finetune"]["phases"][0]
    first_phase["settings"].update(settings or {})
    first_phase["lora"].update(lora or {})
    first_phase.update(phase or {})
    request["finetune"].update(finetune)
    return request


def without(request, *path):
    *parents, key = path
    node = request
    for parent in parents:
        node = node[parent]
    del node[key]
    return request


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


def test_schema_publishes_the_pipeline_request_and_its_basic_values(logged_in_api):
    response = logged_in_api.get(api_paths.SCHEMA)

    assert response.status_code == 200
    schema = response.json()
    assert set(schema["properties"]) >= {"name", "finetune"}
    assert "learning_rate" in schema["$defs"]["SftSettings"]["required"]


def test_the_standard_request_is_published_and_valid(validate, logged_in_api):
    standard = logged_in_api.get(api_paths.STANDARD_PIPELINE).json()

    assert standard["finetune"]["phases"][0]["settings"]["learning_rate"] == 1e-4
    assert validate(standard).status_code == 200


def test_a_valid_request_comes_back_resolved(validate):
    response = validate(pipeline_request(settings={"warmup_steps": 10}))

    assert response.status_code == 200, response.text
    finetune = response.json()["request"]["finetune"]
    assert finetune["base_model"] == f"hf:{BASE_MODEL}@{COMMIT}"
    assert finetune["phases"][0]["settings"]["warmup_steps"] == 10


def test_the_hf_token_secret_is_used_to_look_up_the_base_model(validate, hugging_face):
    validate(pipeline_request(), secrets={"hf_token": HF_TOKEN})

    assert hugging_face.lookups == [(BASE_MODEL, "main", HF_TOKEN)]


@pytest.mark.parametrize(
    "request_",
    [
        {**pipeline_request(), "gpus": 2},
        pipeline_request(phase={"epochs": 3}),
        pipeline_request(gpus=1),
    ],
)
def test_unknown_fields_are_rejected(validate, request_):
    assert "Extra inputs are not permitted" in rejection(validate(request_))


@pytest.mark.parametrize(
    "path",
    [
        ("finetune", "base_model"),
        ("finetune", "phases", 0, "dataset"),
        ("finetune", "phases", 0, "settings", "learning_rate"),
        ("finetune", "phases", 0, "settings", "num_train_epochs"),
        ("finetune", "phases", 0, "lora", "r"),
    ],
)
def test_requests_missing_a_basic_value_are_rejected(validate, path):
    detail = validate(without(pipeline_request(), *path)).json()["detail"]

    assert detail == [{"loc": ["body", "request", *path], "msg": "Field required"}]


def test_settings_unknown_to_the_trl_config_are_rejected(validate):
    message = rejection(validate(pipeline_request(settings={"lerning_rate": 1e-4})))

    assert "lerning_rate" in message
    assert "SFTConfig" in message


@pytest.mark.parametrize("setting", ["num_train_epochs", "warmup_steps"])
def test_settings_of_the_wrong_type_are_rejected(validate, setting):
    assert setting in rejection(validate(pipeline_request(settings={setting: "x"})))


def test_trust_remote_code_is_rejected_wherever_it_appears(validate):
    settings = {"model_init_kwargs": {"trust_remote_code": True}}

    message = rejection(validate(pipeline_request(settings=settings)))

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
        pipeline_request(settings={"run_name": f"run-{HF_TOKEN}"}),
        secrets={"hf_token": HF_TOKEN},
    )

    assert "Secret" in rejection(response)
    assert HF_TOKEN not in response.text


def test_a_hugging_face_token_inside_the_request_is_rejected_without_secrets(validate):
    response = validate(pipeline_request(settings={"run_name": HF_TOKEN}))

    assert "Secret" in rejection(response)


def test_every_error_is_reported_with_its_path(validate):
    response = validate(pipeline_request(settings={"lerning_rate": 1}, lora={"rank": 8}))

    detail = response.json()["detail"]

    assert [error["loc"][-1] for error in detail] == ["lerning_rate", "rank"]


def test_validation_requires_login(api, hugging_face):
    response = api.post(api_paths.VALIDATE_PIPELINE, json={"request": pipeline_request()})

    assert response.status_code == 401


@pytest.mark.parametrize("schema_file", [None, "not json"])
def test_settings_go_unchecked_without_a_usable_trainer_config_schema(
    validate, tmp_path, monkeypatch, schema_file
):
    if schema_file:
        (tmp_path / "SFTConfig.json").write_text(schema_file)
    monkeypatch.setattr(config, "TRAINER_CONFIGS_DIRECTORY", tmp_path)

    response = validate(pipeline_request(settings={"anything": "goes"}, lora={"new_option": 1}))

    assert response.status_code == 200, response.text
