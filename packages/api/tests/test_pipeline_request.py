import hashlib
import json

import pytest
from sqlalchemy import create_engine

from mlp_api.datasets.registry import delete_dataset_version, upload_dataset_version
from mlp_api.pipelines.hugging_face import HubModel
from mlp_api.pipelines.pipeline_request import validate_pipeline_request
from mlp_api.storage.database import create_tables
from mlp_core import config
from mlp_core.pipeline_request.schema import Phase

from .conftest import FakeHuggingFace, FakeModelRegistry, FakeObjectStore
from .test_models import register

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
COMMIT = "7ae557604adf67be50417f59c2c2f167def9a775"
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
def model_registry():
    return FakeModelRegistry()


@pytest.fixture
def object_store():
    return FakeObjectStore()


@pytest.fixture
def validate(hugging_face, engine, chat_dataset, model_registry, object_store):
    def run(request, secrets=None):
        return validate_pipeline_request(
            request, secrets or {}, hugging_face, engine, model_registry, object_store
        )

    return run


def starting_from(model: str) -> dict:
    """A valid request that starts from a Model Version instead of a Base Model."""
    request = without(pipeline_request(), "finetune", "base_model")
    request["finetune"]["from"] = model
    return request


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
        {**pipeline_request(), "serve": {}},
        pipeline_request(phase={"epochs": 3}),
        pipeline_request(gpus=1),
    ],
)
def test_unknown_fields_are_rejected(validate, request_):
    assert "Extra inputs are not permitted" in rejection(validate(request_))


def test_a_phase_without_a_dataset_is_rejected(validate):
    path = ("finetune", "phases", 0, "dataset")

    _, errors = validate(without(pipeline_request(), *path))

    assert errors == [{"loc": list(path), "msg": "Field required"}]


@pytest.mark.parametrize(
    ("algorithm", "dataset"),
    [("sft", "dataset:chat"), ("dpo", "dataset:pairs"), ("kto", "dataset:labels")],
)
def test_a_phase_of_only_algorithm_and_dataset_trains_with_its_algorithms_defaults(
    validate, preference_datasets, algorithm, dataset
):
    request = pipeline_request()
    request["finetune"]["phases"] = [{"algorithm": algorithm, "dataset": dataset}]

    resolved, errors = validate(request)

    assert errors == []
    [phase] = resolved.finetune.phases
    defaults = config.ALGORITHMS[algorithm]
    assert phase.settings.model_dump() == defaults["default_settings"]
    assert phase.lora.model_dump() == defaults["default_lora"]


def test_named_settings_replace_only_their_defaults(validate):
    resolved, _ = validate(pipeline_request(settings={"learning_rate": 1e-5}, lora={"r": 8}))

    [phase] = resolved.finetune.phases
    assert phase.settings.learning_rate == 1e-5
    assert phase.settings.lr_scheduler_type == "cosine"
    assert (phase.lora.r, phase.lora.lora_alpha) == (8, 32)


def test_a_full_phase_gets_no_lora(validate):
    request = without(pipeline_request(phase={"method": "full"}), "finetune", "phases", 0, "lora")

    resolved, _ = validate(request)

    assert resolved.finetune.phases[0].lora is None


def test_rl_phases_default_to_no_lora_dropout():
    phase = Phase.model_validate(
        {"algorithm": "grpo", "dataset": "dataset:chat", "rewards": REWARDS, "lora": {"r": 8}}
    )

    assert phase.lora.lora_dropout == 0.0


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


@pytest.mark.parametrize(
    ("block", "setting", "value"),
    [
        ("settings", "output_dir", "/tmp/out"),
        ("settings", "save_steps", 100),
        ("settings", "report_to", "wandb"),
        ("settings", "push_to_hub", True),
        ("settings", "model_init_kwargs", {"trust_remote_code": True}),
        ("settings", "trust_remote_code", True),
        ("lora", "task_type", "SEQ_CLS"),
    ],
)
def test_blocked_settings_are_rejected(validate, block, setting, value):
    message = rejection(validate(pipeline_request(**{block: {setting: value}})))

    assert f"'{block}', '{setting}']: `{setting}` is blocked" in message


UNSERVABLE_ADAPTERS = [
    ({"use_dora": True}, "DoRA Adapters"),
    ({"modules_to_save": ["lm_head"]}, "modules_to_save"),
    ({"bias": "all"}, "a trained bias"),
    ({"r": 1024}, "rank 1024"),
]


@pytest.mark.parametrize(("lora", "problem"), UNSERVABLE_ADAPTERS)
def test_an_adapter_vllm_cannot_serve_is_rejected_when_kept_as_an_adapter(validate, lora, problem):
    message = rejection(validate(pipeline_request(lora=lora)))

    assert "0, 'lora']: " in message
    assert problem in message and "`output: merged`" in message


@pytest.mark.parametrize("method", ["lora", "qlora"])
@pytest.mark.parametrize(("lora", "problem"), UNSERVABLE_ADAPTERS)
def test_a_merged_adapter_is_plain_weights_so_vllms_adapter_rules_do_not_apply(
    validate, method, lora, problem
):
    request = pipeline_request(lora=lora, phase={"method": method, "output": "merged"})

    resolved, errors = validate(request)

    assert errors == []
    assert resolved.finetune.phases[0].output == "merged"


def test_a_phase_trains_a_lora_adapter_by_default(validate):
    request = without(pipeline_request(), "finetune", "phases", 0, "method")

    resolved, errors = validate(request)

    assert errors == []
    assert (resolved.finetune.phases[0].method, resolved.finetune.phases[0].output) == (
        "lora",
        "adapter",
    )


@pytest.mark.parametrize(
    ("method", "lora"),
    [("qlora", {}), ("lora", {"use_rslora": True}), ("qlora", {"use_rslora": True})],
)
def test_qlora_and_rslora_phases_are_valid(validate, method, lora):
    _, errors = validate(pipeline_request(lora=lora, phase={"method": method}))

    assert errors == []


def test_a_full_phase_trains_every_weight_so_it_takes_no_lora(validate):
    full = pipeline_request(phase={"method": "full"})

    assert "0, 'lora']: `full` trains every weight" in rejection(validate(full))
    _, errors = validate(without(full, "finetune", "phases", 0, "lora"))
    assert errors == []


def test_a_dataset_the_algorithm_cannot_train_on_is_rejected(validate, engine, tmp_path):
    path = tmp_path / "preferences.jsonl"
    path.write_text('{"chosen": "yes", "rejected": "no"}\n')
    upload_dataset_version(engine, FakeObjectStore(), "preferences", path)

    message = rejection(validate(pipeline_request(phase={"dataset": "dataset:preferences"})))

    assert "`preferences@1` has preference rows; sft trains on" in message


def upload_rows(engine, tmp_path, name: str, row: str) -> None:
    path = tmp_path / f"{name}.jsonl"
    path.write_text(row + "\n")
    upload_dataset_version(engine, FakeObjectStore(), name, path)


def then(algorithm: str, dataset: str) -> dict:
    """A later Phase, which continues the Adapter and so has no `lora`."""
    first = pipeline_request()["finetune"]["phases"][0]
    return {**without(first, "lora"), "algorithm": algorithm, "dataset": dataset}


@pytest.fixture
def preference_datasets(engine, tmp_path):
    upload_rows(engine, tmp_path, "pairs", '{"prompt": "Hi", "chosen": "Hi!", "rejected": "Go."}')
    upload_rows(engine, tmp_path, "labels", '{"prompt": "Hi", "completion": "Go.", "label": false}')


def test_phases_chain_each_on_its_own_pinned_dataset(validate, preference_datasets):
    request = pipeline_request()
    request["finetune"]["phases"] += [then("dpo", "dataset:pairs"), then("kto", "dataset:labels")]

    resolved, errors = validate(request)

    assert errors == []
    phases = resolved.finetune.phases
    assert [phase.algorithm for phase in phases] == ["sft", "dpo", "kto"]
    assert [phase.dataset for phase in phases] == [
        "dataset:chat@2",
        "dataset:pairs@1",
        "dataset:labels@1",
    ]


@pytest.mark.parametrize(
    ("algorithm", "dataset", "message"),
    [
        ("dpo", "dataset:chat", "`chat@2` has messages rows; dpo trains on preference rows"),
        ("kto", "dataset:pairs", "`pairs@1` has preference rows; kto trains on unpaired_pref"),
    ],
)
def test_a_dataset_of_the_wrong_row_format_for_a_later_phase_is_rejected(
    validate, preference_datasets, algorithm, dataset, message
):
    request = pipeline_request()
    request["finetune"]["phases"].append(then(algorithm, dataset))

    assert f"1, 'dataset']: {message}" in rejection(validate(request))


def test_the_first_phase_trains_a_new_adapter_with_the_lora_defaults(validate):
    resolved, _ = validate(without(pipeline_request(), "finetune", "phases", 0, "lora"))

    assert resolved.finetune.phases[0].lora.model_dump() == config.LORA_DEFAULTS


def test_a_later_phase_continues_the_adapter_so_it_takes_no_lora(validate, preference_datasets):
    request = pipeline_request()
    later = then("dpo", "dataset:pairs")
    later["lora"] = request["finetune"]["phases"][0]["lora"]
    request["finetune"]["phases"].append(later)

    message = rejection(validate(request))

    assert "1, 'lora']: continues the Adapter of Phase 1" in message
    assert "`output: merged` on Phase 1" in message


def test_a_full_phase_after_an_adapter_merges_it_first(validate, preference_datasets):
    request = pipeline_request()
    request["finetune"]["phases"].append({**then("dpo", "dataset:pairs"), "method": "full"})

    _, errors = validate(request)

    assert errors == []


@pytest.mark.parametrize("earlier", [{"output": "merged"}, {"method": "full"}])
def test_a_phase_after_full_weights_trains_a_new_adapter_with_the_lora_defaults(
    validate, preference_datasets, earlier
):
    request = pipeline_request(phase=earlier)
    if earlier.get("method") == "full":
        without(request, "finetune", "phases", 0, "lora")
    request["finetune"]["phases"].append(then("dpo", "dataset:pairs"))

    resolved, errors = validate(request)

    assert errors == []
    assert resolved.finetune.phases[1].lora.model_dump() == config.LORA_DEFAULTS


def test_settings_of_a_later_phase_are_checked_against_its_algorithms_config(
    validate, preference_datasets
):
    request = pipeline_request()
    request["finetune"]["phases"].append(
        {**then("dpo", "dataset:pairs"), "settings": {**then("dpo", "")["settings"], "bta": 0.1}}
    )

    assert "1, 'settings', 'bta']: `bta` is not a DPOConfig" in rejection(validate(request))


def test_a_full_weight_model_version_to_start_from_resolves_to_its_latest_version(
    validate, model_registry
):
    for version in (1, 2):
        register(model_registry, FakeObjectStore(), "uploaded", version, weights="full")

    request, errors = validate(starting_from("model:uploaded"))

    assert errors == []
    assert request.finetune.base_model is None
    assert request.model_dump(mode="json")["finetune"]["from"] == "model:uploaded@2"


@pytest.mark.parametrize(
    ("model", "message"),
    [
        ("model:missing", "no Registered Model `missing`"),
        ("model:uploaded@3", "`uploaded` has no version 3"),
        ("model:qwen-sft@1", "is an Adapter"),
    ],
)
def test_a_model_version_to_start_from_must_exist_and_hold_full_weights(
    validate, model_registry, model, message
):
    register(model_registry, FakeObjectStore(), "uploaded", 1, weights="full")
    register(model_registry, FakeObjectStore(), "qwen-sft", 1)

    assert message in rejection(validate(starting_from(model)))


@pytest.mark.parametrize("both", [False, True], ids=["neither", "both"])
def test_finetune_starts_from_exactly_one_model(validate, model_registry, both):
    register(model_registry, FakeObjectStore(), "uploaded", 1, weights="full")
    request = starting_from("model:uploaded@1")
    if both:
        request["finetune"]["base_model"] = f"hf:{BASE_MODEL}"
    else:
        del request["finetune"]["from"]

    assert "exactly one" in rejection(validate(request))


MARKED_TEMPLATE = (
    "{% for m in messages %}{% if m.role == 'assistant' %}{%- generation %}{{ m.content }}"
    "{% endgeneration %}{% else %}{{ m.content }}{% endif %}{% endfor %}"
)
UNMARKED_TEMPLATE = "{% for m in messages %}{{ m.content }}{% endfor %}"
ASSISTANT_ONLY = {"assistant_only_loss": True}


@pytest.mark.parametrize(
    "files",
    [
        {"chat_template.jinja": MARKED_TEMPLATE.encode()},
        {"tokenizer_config.json": json.dumps({"chat_template": MARKED_TEMPLATE}).encode()},
    ],
    ids=["jinja-file", "tokenizer-config"],
)
def test_assistant_only_loss_trains_with_a_template_that_marks_assistant_turns(
    validate, hugging_face, files
):
    hugging_face.files[BASE_MODEL] = files

    request, errors = validate(pipeline_request(settings=ASSISTANT_ONLY))

    assert errors == []
    assert request.finetune.phases[0].settings.assistant_only_loss is True


def test_assistant_only_loss_is_rejected_for_a_template_without_generation_markers(
    validate, hugging_face
):
    hugging_face.files[BASE_MODEL] = {"chat_template.jinja": UNMARKED_TEMPLATE.encode()}

    message = rejection(validate(pipeline_request(settings=ASSISTANT_ONLY)))

    assert "'settings', 'assistant_only_loss']" in message
    assert "doesn't mark the assistant's turns with `{% generation %}`" in message


def test_a_template_trl_swaps_for_a_training_template_needs_no_markers(
    validate, hugging_face, tmp_path, monkeypatch
):
    # TRL brings marked versions of known templates, e.g. Qwen2.5's, and trains with those.
    known = tmp_path / "TrainingChatTemplates.json"
    known.write_text(json.dumps([hashlib.sha256(UNMARKED_TEMPLATE.encode()).hexdigest()]))
    monkeypatch.setattr(config, "TRAINING_CHAT_TEMPLATES", known)
    hugging_face.files[BASE_MODEL] = {"chat_template.jinja": UNMARKED_TEMPLATE.encode()}

    _, errors = validate(pipeline_request(settings=ASSISTANT_ONLY))

    assert errors == []


def test_assistant_only_loss_is_rejected_for_a_model_without_a_chat_template(
    validate, hugging_face
):
    hugging_face.files[BASE_MODEL] = {"tokenizer_config.json": b"{}"}

    assert "has no chat template" in rejection(validate(pipeline_request(settings=ASSISTANT_ONLY)))


def test_the_template_of_a_model_version_to_start_from_is_read_from_its_files(
    validate, model_registry, object_store
):
    files = {"model.safetensors": b"weights", "chat_template.jinja": UNMARKED_TEMPLATE.encode()}
    register(model_registry, object_store, "uploaded", 1, files=files, weights="full")
    request = starting_from("model:uploaded@1")
    request["finetune"]["phases"][0]["settings"].update(ASSISTANT_ONLY)

    assert "model:uploaded@1's chat template doesn't mark" in rejection(validate(request))


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ('{"prompt": "Hi", "completion": "Hello"}', "prompt_completion rows train only on"),
        ('{"text": "Hi"}', "is for messages rows"),
    ],
)
def test_assistant_only_loss_is_only_for_messages_rows(validate, engine, tmp_path, row, message):
    upload_rows(engine, tmp_path, "rows", row)
    request = pipeline_request(settings=ASSISTANT_ONLY, phase={"dataset": "dataset:rows"})

    assert message in rejection(validate(request))


def test_assistant_only_loss_is_only_an_sft_setting(validate, preference_datasets):
    request = pipeline_request()
    request["finetune"]["phases"].append(then("dpo", "dataset:pairs"))
    request["finetune"]["phases"][1]["settings"].update(ASSISTANT_ONLY)

    assert "`assistant_only_loss` is not a DPOConfig setting" in rejection(validate(request))


def test_a_phase_leaving_assistant_only_loss_out_passes_trl_no_value_for_it(validate):
    request, errors = validate(pipeline_request())

    assert errors == []
    assert "assistant_only_loss" not in request.finetune.phases[0].settings.model_dump()


def test_finetune_trains_on_the_hf_backend_by_default(validate):
    request, errors = validate(without(pipeline_request(), "finetune", "backend"))

    assert errors == []
    assert request.finetune.backend == "hf"


@pytest.mark.parametrize("method", ["lora", "qlora", "full"])
def test_sft_trains_with_every_method_on_unsloth(validate, method):
    phase = {"method": method, **({"lora": None} if method == "full" else {})}
    request = pipeline_request(phase=phase, backend="unsloth")

    assert validate(request)[1] == []


REWARDS = {"exact": {"weight": 1.0, "source": "def reward(sample, item):\n    return 1.0\n"}}


@pytest.mark.parametrize(
    ("phase", "message"),
    [
        (
            {"algorithm": "distillation", "teacher": f"hf:{BASE_MODEL}"},
            "unsloth doesn't train distillation Phases; use backend `hf`",
        ),
        (
            {"algorithm": "grpo", "method": "full", "lora": None, "rewards": REWARDS},
            "unsloth trains grpo Phases with lora, qlora, not full; use backend `hf`",
        ),
    ],
)
def test_an_algorithm_or_method_the_backend_lacks_is_rejected(validate, phase, message):
    request = pipeline_request(phase=phase, backend="unsloth")

    assert message in rejection(validate(request))


def test_assistant_only_loss_on_unsloth_needs_no_generation_markers(validate, hugging_face):
    # Unsloth finds the assistant's turns from the rendered template itself.
    hugging_face.files[BASE_MODEL] = {"chat_template.jinja": UNMARKED_TEMPLATE.encode()}

    _, errors = validate(pipeline_request(settings=ASSISTANT_ONLY, backend="unsloth"))

    assert errors == []
