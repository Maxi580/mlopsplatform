import pytest

from mlp_api.pipelines.cluster import KubeflowRun
from mlp_api.pipelines.reconciler import reconcile_pipelines
from mlp_core import api_paths

from .test_datasets import jsonl, upload
from .test_distill_stage import (  # noqa: F401
    DISTILLED,
    distill_step,
    distilling_into_finetune,
    prompts,
)
from .test_evaluate_stage import tasks
from .test_models import delete, register
from .test_pipeline_request import pipeline_request, then
from .test_pipelines import pipeline, submit, submittable  # noqa: F401

pytestmark = pytest.mark.usefixtures("submittable")

CHECKPOINT = b"optimizer state"


def two_phases() -> dict:
    request = pipeline_request()
    request["finetune"]["phases"].append(then("dpo", "dataset:pairs"))
    return request


@pytest.fixture
def pairs(logged_in_api):
    upload(logged_in_api, "pairs", jsonl({"prompt": "Hi", "chosen": "Hello!", "rejected": "Go."}))


def end(api, pipeline_id, state):
    """Puts the Pipeline's Kubeflow run into `state` and reconciles."""
    cluster = api.app.state.cluster
    run_id = pipeline(api, pipeline_id).json()["kubeflow_run_url"].rsplit("/", 1)[-1]
    cluster.runs[run_id] = KubeflowRun(state, None)
    reconcile_pipelines(api.app.state.engine, cluster)


def failed_pipeline(api, request=None) -> int:
    pipeline_id = submit(api, request or two_phases()).json()["id"]
    end(api, pipeline_id, "FAILED")
    return pipeline_id


def registered_phase(api, pipeline_id, phase_index, version):
    """The Model Version the Phase registered, as the finetune Stage tags it."""
    state = api.app.state
    register(state.model_registry, state.object_store, "qwen-sft", version, pipeline_id)
    state.model_registry.versions[-1].tags["phase"] = str(phase_index + 1)


def saved_checkpoint(api, pipeline_id, phase_index):
    key = f"checkpoints/{pipeline_id}/{phase_index}/checkpoint-5/trainer_state.json"
    api.app.state.object_store.objects[key] = CHECKPOINT


def resume(api, pipeline_id, secrets=None):
    path = api_paths.RESUME_PIPELINE.format(id=pipeline_id)
    return api.post(path, json={"secrets": secrets or {}})


def parameters(task) -> dict:
    """Each input parameter's constant, or the task whose output it is handed."""
    return {
        name: p["runtimeValue"]["constant"]
        if "runtimeValue" in p
        else p["taskOutputParameter"]["producerTask"]
        for name, p in task["inputs"]["parameters"].items()
    }


def test_a_phase_saves_checkpoints_at_the_platforms_interval_and_resumes_none(
    logged_in_api, cluster
):
    submit(logged_in_api)

    finetune = parameters(tasks(cluster)["finetune"])
    # checkpoint_minutes from the settings ConfigMap.
    assert finetune["checkpoint_minutes"] == "30"
    assert finetune["resume_checkpoint"] == ""
    assert finetune["stop_after_checkpoint"] == ""


def test_resume_skips_finished_phases_and_continues_the_failed_one_from_its_checkpoint(
    logged_in_api, pairs, cluster
):
    failed = failed_pipeline(logged_in_api)
    registered_phase(logged_in_api, failed, 0, 1)
    saved_checkpoint(logged_in_api, failed, 1)

    response = resume(logged_in_api, failed)

    assert response.status_code == 202, response.text
    resumed = response.json()["id"]
    assert set(tasks(cluster)) == {"fetch", "finetune"}
    finetune = parameters(tasks(cluster)["finetune"])
    assert finetune["pipeline_id"] == str(resumed)
    assert finetune["phase_index"] == "1"
    assert finetune["previous_model_version"] == "model:qwen-sft@1"
    assert finetune["resume_checkpoint"] == f"checkpoints/{failed}/1/"
    stored = pipeline(logged_in_api, resumed).json()
    assert stored["resumed_from"] == failed
    assert stored["request"] == pipeline(logged_in_api, failed).json()["request"]


def test_a_phase_that_saved_no_checkpoint_starts_afresh(logged_in_api, pairs, cluster):
    failed = failed_pipeline(logged_in_api)

    assert resume(logged_in_api, failed).status_code == 202

    sft, dpo = tasks(cluster)["finetune"], tasks(cluster)["finetune-2"]
    assert parameters(sft)["resume_checkpoint"] == ""
    assert parameters(dpo)["previous_model_version"] == "finetune"


def test_resuming_a_resumed_pipeline_reuses_what_both_made(logged_in_api, pairs, cluster):
    request = two_phases()
    request["finetune"]["phases"].append(then("dpo", "dataset:pairs"))
    first = failed_pipeline(logged_in_api, request)
    registered_phase(logged_in_api, first, 0, 1)
    second = resume(logged_in_api, first).json()["id"]
    registered_phase(logged_in_api, second, 1, 2)
    saved_checkpoint(logged_in_api, second, 2)
    end(logged_in_api, second, "CANCELED")

    third = resume(logged_in_api, second).json()["id"]

    finetune = parameters(tasks(cluster)["finetune"])
    assert finetune["pipeline_id"] == str(third)
    assert finetune["phase_index"] == "2"
    assert finetune["previous_model_version"] == "model:qwen-sft@2"
    assert finetune["resume_checkpoint"] == f"checkpoints/{second}/2/"


def test_a_checkpoint_the_resumed_pipeline_made_none_of_is_carried_on(
    logged_in_api, pairs, cluster
):
    first = failed_pipeline(logged_in_api)
    saved_checkpoint(logged_in_api, first, 0)
    second = resume(logged_in_api, first).json()["id"]
    end(logged_in_api, second, "FAILED")

    resume(logged_in_api, second)

    assert parameters(tasks(cluster)["finetune"])["resume_checkpoint"] == f"checkpoints/{first}/0/"


def test_when_every_phase_finished_evaluate_gets_the_last_model_version(logged_in_api, cluster):
    request = {**pipeline_request(), "evaluate": {"benchmarks": ["lm_eval:gsm8k"], "limit": 2}}
    failed = failed_pipeline(logged_in_api, request)
    registered_phase(logged_in_api, failed, 0, 1)

    assert resume(logged_in_api, failed).status_code == 202

    assert "finetune" not in tasks(cluster)
    evaluated = parameters(tasks(cluster)["evaluate"])["request"]
    assert '"model":"model:qwen-sft@1"' in evaluated.replace(" ", "")


@pytest.mark.usefixtures("prompts")
def test_a_finished_distill_is_not_run_again(logged_in_api, cluster):
    failed = submit(logged_in_api, distilling_into_finetune()).json()["id"]
    distill_step(logged_in_api, failed, jsonl(DISTILLED))
    end(logged_in_api, failed, "FAILED")

    assert resume(logged_in_api, failed).status_code == 202

    assert "distill" not in tasks(cluster)
    finetune = parameters(tasks(cluster)["finetune"])
    assert finetune["distilled_dataset"] == "dataset:qwen-sft@1"


def test_the_resumed_pipeline_gets_the_fresh_secrets(logged_in_api, pairs, cluster):
    failed = failed_pipeline(logged_in_api)

    resumed = resume(logged_in_api, failed, {"hf_token": "hf_" + "n3w" * 12}).json()["id"]

    assert cluster.secrets[f"pipeline-{resumed}"] == {"hf_token": "hf_" + "n3w" * 12}


def test_a_running_or_succeeded_pipeline_cannot_be_resumed(logged_in_api, pairs):
    running = submit(logged_in_api, two_phases()).json()["id"]

    response = resume(logged_in_api, running)

    assert response.status_code == 409
    assert "failed or cancelled" in response.json()["detail"]
    end(logged_in_api, running, "SUCCEEDED")
    assert resume(logged_in_api, running).status_code == 409


def test_resuming_an_unknown_pipeline_is_not_found(logged_in_api):
    assert resume(logged_in_api, 42).status_code == 404


def test_a_model_version_a_running_resume_reuses_cannot_be_deleted(logged_in_api, pairs):
    failed = failed_pipeline(logged_in_api)
    registered_phase(logged_in_api, failed, 0, 1)
    resumed = resume(logged_in_api, failed).json()["id"]

    response = delete(logged_in_api, "qwen-sft", 1)

    assert response.status_code == 409
    assert f"(#{resumed})" in response.json()["detail"]


def checkpoints(api) -> list[dict]:
    response = api.get(api_paths.CHECKPOINTS)
    assert response.status_code == 200, response.text
    return response.json()


def delete_checkpoint(api, pipeline_id, phase_index):
    path = api_paths.CHECKPOINT.format(pipeline_id=pipeline_id, phase_index=phase_index)
    return api.delete(path)


def test_kept_checkpoints_are_listed_with_their_pipeline_and_size(logged_in_api, pairs):
    failed = failed_pipeline(logged_in_api)
    saved_checkpoint(logged_in_api, failed, 1)

    assert checkpoints(logged_in_api) == [
        {
            "pipeline_id": failed,
            "pipeline_name": "qwen-sft",
            "phase_index": 1,
            "size_bytes": len(CHECKPOINT),
        }
    ]


def test_a_checkpoint_can_be_deleted(logged_in_api, pairs, object_store):
    failed = failed_pipeline(logged_in_api)
    saved_checkpoint(logged_in_api, failed, 1)

    assert delete_checkpoint(logged_in_api, failed, 1).status_code == 204

    assert checkpoints(logged_in_api) == []
    assert object_store.objects_under("platform", "checkpoints/") == {}


def test_deleting_a_checkpoint_a_running_resume_needs_is_refused(logged_in_api, pairs):
    failed = failed_pipeline(logged_in_api)
    registered_phase(logged_in_api, failed, 0, 1)
    saved_checkpoint(logged_in_api, failed, 1)
    resumed = resume(logged_in_api, failed).json()["id"]

    response = delete_checkpoint(logged_in_api, failed, 1)

    assert response.status_code == 409
    assert f"qwen-sft (#{resumed})" in response.json()["detail"]
    end(logged_in_api, resumed, "SUCCEEDED")
    assert delete_checkpoint(logged_in_api, failed, 1).status_code == 204


def test_the_checkpoint_of_a_running_phase_cannot_be_deleted(logged_in_api, pairs):
    running = submit(logged_in_api, two_phases()).json()["id"]
    saved_checkpoint(logged_in_api, running, 0)

    assert delete_checkpoint(logged_in_api, running, 0).status_code == 409


def test_deleting_a_missing_checkpoint_is_not_found(logged_in_api):
    assert delete_checkpoint(logged_in_api, 7, 0).status_code == 404


def test_a_running_pipelines_checkpoints_are_not_listed(logged_in_api, pairs):
    running = submit(logged_in_api, two_phases()).json()["id"]
    saved_checkpoint(logged_in_api, running, 0)

    assert checkpoints(logged_in_api) == []
