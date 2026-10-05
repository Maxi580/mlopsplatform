import pytest

from mlp_api.pipelines.hugging_face import HubModel

from .test_datasets import jsonl, upload
from .test_endpoints import register_adapter
from .test_evaluate_stage import errors, inputs, tasks, validate
from .test_pipeline_request import BASE_MODEL, COMMIT, pipeline_request
from .test_pipelines import submit, submittable  # noqa: F401

pytestmark = pytest.mark.usefixtures("submittable", "prompts")

PINNED_BASE_MODEL = f"hf:{BASE_MODEL}@{COMMIT}"
TEACHER = "Qwen/Qwen2.5-7B-Instruct"
TEACHER_COMMIT = "7eac4e5"


@pytest.fixture
def prompts(logged_in_api, hugging_face):
    """The `prompts` Dataset, and a larger Qwen on the hub to teach."""
    hugging_face.models[TEACHER] = HubModel(TEACHER_COMMIT, needs_remote_code=False)
    upload(logged_in_api, "prompts", jsonl({"prompt": "Why is the sky blue?"}))


def distilling(teacher=f"hf:{TEACHER}", **phase) -> dict:
    """A request whose one Phase learns the Teacher's token probabilities on `prompts`."""
    settings = {
        "learning_rate": 1e-4,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "max_completion_length": 256,
    }
    distillation = {"algorithm": "distillation", "dataset": "dataset:prompts", "teacher": teacher}
    return pipeline_request(phase={**distillation, "settings": settings, **phase})


def resolved_phase(api, request) -> dict:
    response = validate(api, request)
    assert response.status_code == 200, response.text
    return response.json()["request"]["finetune"]["phases"][0]


def test_the_teacher_and_the_prompts_are_pinned(logged_in_api):
    phase = resolved_phase(logged_in_api, distilling())

    assert phase["teacher"] == f"hf:{TEACHER}@{TEACHER_COMMIT}"
    assert phase["dataset"] == "dataset:prompts@1"


def test_a_model_version_can_teach(logged_in_api, model_registry, object_store):
    register_adapter(model_registry, object_store, "qwen-sft", PINNED_BASE_MODEL)

    assert resolved_phase(logged_in_api, distilling("model:qwen-sft"))["teacher"] == (
        "model:qwen-sft@1"
    )


@pytest.mark.parametrize("teacher", ["gpt-4.1", "endpoint:chat"])
def test_a_teacher_outside_the_job_is_rejected(logged_in_api, teacher):
    [error] = errors(logged_in_api, distilling(teacher))

    assert error["loc"] == ["finetune", "phases", 0]
    assert "loaded beside the Student: hf:… or model:…" in error["msg"]


def test_an_api_teacher_is_rejected(logged_in_api):
    request = distilling("gpt-4.1", api_url="https://api.openai.com/v1")

    locs = [error["loc"] for error in errors(logged_in_api, request)]

    assert ["finetune", "phases", 0, "api_url"] in locs


def test_a_distillation_phase_needs_a_teacher(logged_in_api):
    [error] = errors(logged_in_api, distilling(None))

    assert "name the `teacher`" in error["msg"]


def test_only_a_distillation_phase_takes_a_teacher(logged_in_api):
    [error] = errors(logged_in_api, pipeline_request(phase={"teacher": f"hf:{TEACHER}"}))

    assert "a sft Phase learns from no `teacher`" in error["msg"]


def test_a_distillation_phase_trains_on_prompts_only(logged_in_api):
    [error] = errors(logged_in_api, distilling(dataset="dataset:chat"))

    assert "distillation trains on prompt_only rows" in error["msg"]


def test_each_algorithm_requires_its_own_length_setting(logged_in_api):
    request = distilling(settings={**distilling()["finetune"]["phases"][0]["settings"]})
    del request["finetune"]["phases"][0]["settings"]["max_completion_length"]
    sft = pipeline_request()
    del sft["finetune"]["phases"][0]["settings"]["max_length"]

    [distillation_error] = errors(logged_in_api, request)
    [sft_error] = errors(logged_in_api, sft)

    assert distillation_error == {
        "loc": ["finetune", "phases", 0, "settings", "max_completion_length"],
        "msg": "Field required",
    }
    assert sft_error["loc"] == ["finetune", "phases", 0, "settings", "max_length"]


def test_settings_are_checked_against_the_distillation_config(logged_in_api):
    settings = {**distilling()["finetune"]["phases"][0]["settings"], "max_length": 512}

    [error] = errors(logged_in_api, distilling(settings=settings))

    assert error["msg"] == "`max_length` is not a DistillationConfig setting"


@pytest.mark.parametrize("setting", ["teacher_model_name_or_path", "use_vllm"])
def test_the_teacher_is_only_set_by_the_phase(logged_in_api, setting):
    settings = {**distilling()["finetune"]["phases"][0]["settings"], setting: "x"}

    [error] = errors(logged_in_api, distilling(settings=settings))

    assert error["msg"] == f"`{setting}` is blocked; see the README for why"


def test_fetch_pulls_the_teachers_base_model_too(logged_in_api, cluster, model_registry):
    assert submit(logged_in_api, distilling()).status_code == 202

    fetched = inputs(tasks(cluster)["fetch"])["references"]
    assert fetched == f"{PINNED_BASE_MODEL},hf:{TEACHER}@{TEACHER_COMMIT}"
