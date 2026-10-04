import json

import pytest

from mlp_api.pipelines.hugging_face import HubModel
from mlp_api.pipelines.reconciler import reconcile_once
from mlp_core import api_paths

from .test_datasets import datasets, jsonl, upload
from .test_endpoints import register_adapter
from .test_evaluate_stage import container, errors, inputs, running_endpoint, tasks, validate
from .test_pipeline_request import BASE_MODEL, COMMIT, pipeline_request
from .test_pipelines import (  # noqa: F401
    finish_run,
    pipelines,
    submit,
    submittable,
    submitted_pipeline,
)

# Every test may also finetune on the `chat` Dataset, as `submittable` sets up.
pytestmark = pytest.mark.usefixtures("submittable", "prompts")

PINNED_BASE_MODEL = f"hf:{BASE_MODEL}@{COMMIT}"
TEACHER_API_KEY = "sk-teacher-key-0123456789"
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}
DISTILLED = {
    "prompt": [{"role": "user", "content": "Weather in Berlin?"}],
    "completion": [{"role": "assistant", "content": "Sunny."}],
}


@pytest.fixture
def prompts(logged_in_api, hugging_face):
    """The `prompts` Dataset, and the Base Model as a Teacher whose tool calls vLLM can parse."""
    hugging_face.models[BASE_MODEL] = HubModel(COMMIT, needs_remote_code=False, model_type="qwen2")
    upload(logged_in_api, "prompts", jsonl({"prompt": "Weather in Berlin?"}))


def distilling(**distill) -> dict:
    """A request that only distills the `prompts` Dataset with the Base Model as Teacher."""
    return {
        "name": "qwen-distill",
        "distill": {"dataset": "dataset:prompts", "teacher": f"hf:{BASE_MODEL}", **distill},
    }


def distilling_into_finetune(**distill) -> dict:
    request = pipeline_request(phase={"dataset": "@distill"})
    return {**request, "distill": distilling(**distill)["distill"]}


def resolved(api, request, secrets=None) -> dict:
    response = api.post(
        api_paths.VALIDATE_PIPELINE, json={"request": request, "secrets": secrets or {}}
    )
    assert response.status_code == 200, response.text
    return response.json()["request"]


def received_secrets(cluster, pipeline_id) -> list[dict]:
    """What the `distill` step receives from the Pipeline's Secret, each with its variable."""
    _, kubernetes = submitted_pipeline(cluster)
    executor = kubernetes["deploymentSpec"]["executors"]["exec-distill"]
    return [
        mapping
        for secret in executor["secretAsEnv"]
        if secret["secretName"] == f"pipeline-{pipeline_id}"
        for mapping in secret["keyToEnv"]
    ]


def test_the_teacher_and_the_prompts_are_pinned(logged_in_api):
    distill = resolved(logged_in_api, distilling())["distill"]

    assert distill["teacher"] == PINNED_BASE_MODEL
    assert distill["dataset"] == "dataset:prompts@1"


def test_with_tools_the_teachers_tool_parser_is_pinned(logged_in_api):
    distill = resolved(logged_in_api, distilling(tools=[WEATHER]))["distill"]

    assert distill["serving"]["tool_parser"] == "hermes"
    assert distill["tools"] == [
        {**WEATHER, "function": {**WEATHER["function"], "description": None}}
    ]


def test_the_prompts_cannot_share_the_name_of_the_distillation_dataset(logged_in_api):
    [error] = errors(logged_in_api, {**distilling(), "name": "prompts"})

    assert error["loc"] == ["distill", "dataset"]
    assert "rename either" in error["msg"]


def test_tools_need_a_teacher_with_a_tool_parser(logged_in_api, hugging_face):
    hugging_face.models[BASE_MODEL] = HubModel(COMMIT, needs_remote_code=False, model_type="gpt2")

    [error] = errors(logged_in_api, distilling(tools=[WEATHER]))
    overridden = distilling(tools=[WEATHER], serving={"tool_parser": "hermes"})

    assert error["loc"] == ["distill", "tools"]
    assert "no tool parser" in error["msg"]
    assert validate(logged_in_api, overridden).status_code == 200


def test_tool_parameters_must_be_a_json_schema(logged_in_api):
    broken = {"type": "function", "function": {"name": "f", "parameters": {"type": "nope"}}}

    [error] = errors(logged_in_api, distilling(tools=[broken]))

    assert error["loc"] == ["distill", "tools", 0, "function", "parameters"]


def test_tool_names_must_be_unique(logged_in_api):
    [error] = errors(logged_in_api, distilling(tools=[WEATHER, WEATHER]))

    assert error["loc"] == ["distill", "tools"]


def test_the_prompts_must_be_prompt_rows(logged_in_api):
    upload(logged_in_api, "chat2", jsonl({"messages": [{"role": "user", "content": "Hi"}]}))

    [error] = errors(logged_in_api, distilling(dataset="dataset:chat2"))

    assert error["loc"] == ["distill", "dataset"]
    assert "prompt_only" in error["msg"]


@pytest.mark.parametrize(
    ("teacher", "message"),
    [
        ("gpt-4.1", "is hf:…, model:… or endpoint:…"),
        ("model:missing", "no Registered Model `missing`"),
        ("endpoint:chat", "no Endpoint chat is running"),
    ],
)
def test_a_teacher_that_cannot_be_reached_is_rejected(logged_in_api, teacher, message):
    [error] = errors(logged_in_api, distilling(teacher=teacher))

    assert message in error["msg"]


def test_an_adapter_can_teach(logged_in_api, model_registry, object_store):
    register_adapter(model_registry, object_store, "qwen-sft", PINNED_BASE_MODEL)

    assert resolved(logged_in_api, distilling(teacher="model:qwen-sft"))["distill"]["teacher"] == (
        "model:qwen-sft@1"
    )


def test_an_api_teacher_needs_its_key(logged_in_api):
    request = distilling(teacher="gpt-4.1", api_url="https://api.openai.com/v1", tools=[WEATHER])

    [error] = errors(logged_in_api, request)
    secrets = {"teacher_api_key": TEACHER_API_KEY}

    assert error["loc"] == ["distill", "api_url"]
    assert "teacher_api_key" in error["msg"]
    assert resolved(logged_in_api, request, secrets)["distill"]["teacher"] == "gpt-4.1"


def test_distill_output_feeds_finetune(logged_in_api):
    request = resolved(logged_in_api, distilling_into_finetune())

    assert request["finetune"]["phases"][0]["dataset"] == "@distill"


def test_distill_output_needs_distill(logged_in_api):
    [error] = errors(logged_in_api, pipeline_request(phase={"dataset": "@distill"}))

    assert error["loc"] == ["finetune", "phases", 0, "dataset"]
    assert "`distill` isn't enabled" in error["msg"]


def test_distill_runs_on_the_teachers_gpus_and_hands_its_dataset_to_finetune(
    logged_in_api, cluster
):
    pipeline_id = submit(logged_in_api, distilling_into_finetune()).json()["id"]

    distill, finetune = tasks(cluster)["distill"], tasks(cluster)["finetune"]
    assert distill["dependentTasks"] == ["fetch"]
    assert inputs(tasks(cluster)["fetch"])["references"] == PINNED_BASE_MODEL
    assert inputs(distill)["gpus"] == "1"
    assert inputs(distill)["teacher_url"] == ""
    step = container(cluster, "distill")
    assert step["command"] == ["mlp-stage", "distill"]
    assert step["resources"]["accelerator"]["resourceCount"] == "1"
    assert {"name": "HF_HUB_OFFLINE", "value": "1"} in step["env"]
    handoff = finetune["inputs"]["parameters"]["distilled_dataset"]["taskOutputParameter"]
    assert handoff == {"producerTask": "distill", "outputParameterKey": "dataset"}
    assert received_secrets(cluster, pipeline_id) == [
        {"secretKey": "step_token", "envVar": "MLP_STEP_TOKEN"}
    ]
    assert "step_token" in cluster.secrets[f"pipeline-{pipeline_id}"]
    assert pipelines(logged_in_api)[0]["stages"] == ["distill", "finetune"]


def test_only_the_distill_step_receives_the_teachers_key(logged_in_api, cluster):
    request = distilling(teacher="gpt-4.1", api_url="https://api.openai.com/v1")
    secrets = {"teacher_api_key": TEACHER_API_KEY}

    pipeline_id = submit(logged_in_api, request, secrets).json()["id"]

    assert set(tasks(cluster)) == {"distill"}
    assert "resources" not in container(cluster, "distill")
    assert inputs(tasks(cluster)["distill"])["teacher_url"] == "https://api.openai.com/v1"
    assert received_secrets(cluster, pipeline_id) == [
        {"secretKey": "step_token", "envVar": "MLP_STEP_TOKEN"},
        {"secretKey": "teacher_api_key", "envVar": "MLP_TEACHER_API_KEY"},
    ]
    assert cluster.secrets[f"pipeline-{pipeline_id}"]["teacher_api_key"] == TEACHER_API_KEY
    pipeline, _ = submitted_pipeline(cluster)
    assert TEACHER_API_KEY not in json.dumps(pipeline)


def test_an_endpoint_teaches_through_its_service_without_a_gpu(logged_in_api, cluster):
    running_endpoint(logged_in_api, cluster)

    submit(logged_in_api, distilling(teacher="endpoint:chat"))

    assert inputs(tasks(cluster)["distill"])["teacher_url"] == "http://endpoint-chat.mlp.svc:8000"
    assert inputs(tasks(cluster)["distill"])["gpus"] == "0"
    assert "resources" not in container(cluster, "distill")


# What the `distill` step does: send the Distillation Dataset with the step token.
def distill_step(api, pipeline_id, content, token=None):
    token = token or api.app.state.cluster.secrets[f"pipeline-{pipeline_id}"]["step_token"]
    return api.post(
        api_paths.DISTILL_PIPELINE.format(id=pipeline_id),
        content=content,
        headers={"authorization": f"Bearer {token}"},
    )


@pytest.fixture
def distilling_pipeline(logged_in_api) -> int:
    pipeline_id = submit(logged_in_api, distilling()).json()["id"]
    logged_in_api.cookies.clear()
    return pipeline_id


def test_the_distill_step_registers_a_dataset_version_named_after_the_pipeline(
    logged_in_api, distilling_pipeline
):
    response = distill_step(logged_in_api, distilling_pipeline, jsonl(DISTILLED))

    assert response.status_code == 201, response.text
    assert response.json() == {"dataset": "dataset:qwen-distill@1"}
    logged_in_api.post(api_paths.LOGIN, json={"password": "correct horse battery staple"})
    [version] = datasets(logged_in_api)["qwen-distill"]
    assert version["row_format"] == "prompt_completion"


def test_a_step_token_only_registers_for_its_own_pipeline(
    logged_in_api, distilling_pipeline, cluster
):
    token = cluster.secrets[f"pipeline-{distilling_pipeline}"]["step_token"]

    response = distill_step(logged_in_api, distilling_pipeline + 1, jsonl(DISTILLED), token)

    assert response.status_code == 401


def test_a_finished_pipeline_registers_no_more(logged_in_api, distilling_pipeline, cluster):
    token = cluster.secrets[f"pipeline-{distilling_pipeline}"]["step_token"]
    finish_run(cluster, "FAILED")
    reconcile_once(logged_in_api.app.state)

    response = distill_step(logged_in_api, distilling_pipeline, jsonl(DISTILLED), token)

    assert response.status_code == 422


def test_malformed_rows_are_rejected(logged_in_api, distilling_pipeline):
    response = distill_step(logged_in_api, distilling_pipeline, b"not json\n")

    assert response.status_code == 422
    assert "line 1" in response.json()["detail"]
