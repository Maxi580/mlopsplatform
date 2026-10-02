import pytest
from sqlalchemy import create_engine

from mlp_api.database import create_tables
from mlp_api.datasets.registry import delete_dataset_version, upload_dataset_version
from mlp_api.pipelines.hugging_face import HubModel
from mlp_api.pipelines.pipeline_request import validate_pipeline_request
from mlp_core import config

from .conftest import BASE_MODEL, COMMIT, FakeHuggingFace, FakeObjectStore

HF_TOKEN = "hf_" + "a1B2c3D4" * 5


def pipeline_request(settings=None, lora=None, phase=None, **finetune):
    """A valid request with the given values added or replaced."""
    request = {
        "name": "qwen-sft",
        "finetune": {
            "base_model": f"hf:{BASE_MODEL}",
            "backend": "hf",
            "phases": [
                {
                    "algorithm": "sft",
                    "dataset": "dataset:chat",
                    "method": "lora",
                    "settings": {
                        "learning_rate": 1e-4,
                        "num_train_epochs": 3,
                        "per_device_train_batch_size": 8,
                        "gradient_accumulation_steps": 1,
                        "max_length": 1024,
                    },
                    "lora": {
                        "r": 16,
                        "lora_alpha": 32,
                        "lora_dropout": 0.05,
                        "target_modules": "all-linear",
                    },
                }
            ],
        },
    }
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
def hugging_face():
    hugging_face = FakeHuggingFace()
    hugging_face.models[BASE_MODEL] = HubModel(commit=COMMIT, needs_remote_code=False)
    return hugging_face


@pytest.fixture
def engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'platform.db'}")
    create_tables(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def chat_dataset(engine, tmp_path):
    """Dataset `chat` with versions 1 and 2."""
    path = tmp_path / "chat.jsonl"
    path.write_text('{"messages": [{"role": "user", "content": "Hi"}]}\n')
    for _ in range(2):
        upload_dataset_version(engine, FakeObjectStore(), "chat", path)


@pytest.fixture
def validate(hugging_face, engine, chat_dataset):
    def run(request, secrets=None):
        return validate_pipeline_request(request, secrets or {}, hugging_face, engine)

    return run


def rejection(result) -> str:
    request, errors = result
    assert request is None and errors
    return " | ".join(f"{error['loc']}: {error['msg']}" for error in errors)


def test_a_valid_request_comes_back_resolved(validate):
    request, errors = validate(pipeline_request(settings={"warmup_steps": 10}))

    assert errors == []
    assert request.finetune.base_model == f"hf:{BASE_MODEL}@{COMMIT}"
    assert request.finetune.phases[0].settings.warmup_steps == 10


def test_a_dataset_name_resolves_to_its_latest_version(validate):
    request, _ = validate(pipeline_request())

    assert request.finetune.phases[0].dataset == "dataset:chat@2"


def test_a_named_dataset_version_is_kept(validate):
    request, _ = validate(pipeline_request(phase={"dataset": "dataset:chat@1"}))

    assert request.finetune.phases[0].dataset == "dataset:chat@1"


@pytest.mark.parametrize(
    ("dataset", "message"),
    [
        ("dataset:missing", "no Dataset `missing`"),
        ("dataset:chat@3", "`chat` has no version 3"),
        ("dataset:chat@0", "`chat` has no version 0"),
    ],
)
def test_unknown_datasets_and_versions_are_rejected(validate, dataset, message):
    assert message in rejection(validate(pipeline_request(phase={"dataset": dataset})))


def test_a_deleted_dataset_version_is_rejected(validate, engine):
    delete_dataset_version(engine, FakeObjectStore(), "chat", 2)

    assert "`chat` has no version 2" in rejection(
        validate(pipeline_request(phase={"dataset": "dataset:chat@2"}))
    )


def test_a_named_revision_is_looked_up_and_pinned(validate, hugging_face):
    request, _ = validate(pipeline_request(base_model=f"hf:{BASE_MODEL}@v1"))

    assert hugging_face.lookups == [(BASE_MODEL, "v1", None)]
    assert request.finetune.base_model == f"hf:{BASE_MODEL}@{COMMIT}"


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
    _, errors = validate(without(pipeline_request(), *path))

    assert errors == [{"loc": list(path), "msg": "Field required"}]


def test_settings_unknown_to_the_trl_config_are_rejected(validate):
    message = rejection(validate(pipeline_request(settings={"lerning_rate": 1e-4})))

    assert "lerning_rate" in message
    assert "SFTConfig" in message


@pytest.mark.parametrize("setting", ["num_train_epochs", "warmup_steps"])
def test_settings_of_the_wrong_type_are_rejected(validate, setting):
    assert setting in rejection(validate(pipeline_request(settings={setting: "x"})))


def test_a_base_model_unreachable_on_hugging_face_is_rejected(validate):
    response = validate(pipeline_request(base_model="hf:nobody/missing-model"))

    assert "nobody/missing-model" in rejection(response)


def test_a_base_model_that_needs_remote_code_is_rejected(validate, hugging_face):
    hugging_face.models["org/custom"] = HubModel(commit=COMMIT, needs_remote_code=True)

    assert "remote code" in rejection(validate(pipeline_request(base_model="hf:org/custom")))


def test_a_secret_value_inside_the_request_is_rejected(validate):
    result = validate(
        pipeline_request(settings={"run_name": f"run-{HF_TOKEN}"}),
        secrets={"hf_token": HF_TOKEN},
    )

    message = rejection(result)
    assert "Secret" in message
    assert HF_TOKEN not in message


def test_a_hugging_face_token_inside_the_request_is_rejected_without_secrets(validate):
    assert "Secret" in rejection(validate(pipeline_request(settings={"run_name": HF_TOKEN})))


def test_every_error_is_reported_with_its_path(validate):
    _, errors = validate(pipeline_request(settings={"lerning_rate": 1}, lora={"rank": 8}))

    assert [error["loc"][-1] for error in errors] == ["lerning_rate", "rank"]


@pytest.mark.parametrize("schema_file", [None, "not json"])
def test_settings_go_unchecked_without_a_usable_trainer_config_schema(
    validate, tmp_path, monkeypatch, schema_file
):
    if schema_file:
        (tmp_path / "SFTConfig.json").write_text(schema_file)
    monkeypatch.setattr(config, "TRAINER_CONFIGS_DIRECTORY", tmp_path)

    _, errors = validate(pipeline_request(settings={"anything": "goes"}, lora={"new_option": 1}))

    assert errors == []
