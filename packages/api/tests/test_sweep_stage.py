import json

import pytest

from mlp_api.auth.session import issue_step_token
from mlp_core import api_paths
from mlp_core.pipeline_request.schema import Sweep

from .test_datasets import jsonl, upload
from .test_evaluate_stage import PINNED_BASE_MODEL, container, errors, tasks, validate
from .test_pipeline_request import BASE_MODEL, pipeline_request, then
from .test_pipelines import pipeline, pipelines, submit, submittable  # noqa: F401
from .test_quantize_stage import resolved
from .test_resume import end, parameters, resume

# Every test submits or validates a request for the Base Model and the `chat` Dataset.
pytestmark = pytest.mark.usefixtures("submittable")

LEARNING_RATE = {"min": 1e-5, "max": 1e-3, "scale": "log"}
BEST = {"parameters": {"settings": {"learning_rate": 3e-4}, "lora": {}}, "objective": 0.42}


def sweep_block(**sweep) -> dict:
    """The finetune request's Phase as a sweep block, searching its learning rate twice."""
    phase = pipeline_request()["finetune"]["phases"][0]
    return {
        **phase,
        "parameters": {"settings": {"learning_rate": LEARNING_RATE}},
        "objective": {"metric": "eval_loss", "goal": "minimize"},
        "trials": 2,
        **sweep,
    }


def sweeping(request=None, **sweep) -> dict:
    """The finetune request, its Phase training with the best parameters of the sweep."""
    request = request or pipeline_request(phase={"params_from": "@sweep"})
    return {**request, "sweep": sweep_block(**sweep)}


def sweep_only(**sweep) -> dict:
    return {"name": "qwen", "sweep": sweep_block(model=f"hf:{BASE_MODEL}", **sweep)}


def report(api, pipeline_id, result=None, token=None):
    """The `sweep` step's report of its best parameters, sent with the step token."""
    token = token or api.app.state.cluster.secrets[f"pipeline-{pipeline_id}"]["step_token"]
    return api.post(
        api_paths.SWEEP_PIPELINE.format(id=pipeline_id),
        json=result or BEST,
        headers={"authorization": f"Bearer {token}"},
    )


@pytest.mark.parametrize(
    ("algorithm", "objective"),
    [("sft", ("eval_loss", "minimize")), ("grpo", ("reward", "maximize"))],
)
def test_sweep_defaults_to_ten_trials_and_its_algorithms_objective(algorithm, objective):
    block = {"algorithm": algorithm, "dataset": "dataset:chat", "parameters": {"settings": {}}}
    block["parameters"]["settings"]["learning_rate"] = LEARNING_RATE
    if algorithm == "grpo":
        block["rewards"] = {"one": {"weight": 1.0, "source": "def reward(s, i): return 1.0"}}

    sweep = Sweep.model_validate(block)

    assert sweep.trials == 10
    assert (sweep.objective.metric, sweep.objective.goal) == objective


def test_sweep_defaults_to_the_starting_model_of_finetune_and_a_held_out_tenth(logged_in_api):
    sweep = resolved(logged_in_api, sweeping())["sweep"]

    assert sweep["model"] == PINNED_BASE_MODEL
    assert sweep["dataset"] == "dataset:chat@1"
    assert (sweep["backend"], sweep["sampler"], sweep["eval_split"]) == ("hf", "tpe", 0.1)
    assert sweep["eval_dataset"] is None


def test_a_sweep_without_finetune_pins_its_own_model(logged_in_api):
    assert resolved(logged_in_api, sweep_only())["sweep"]["model"] == PINNED_BASE_MODEL


def test_a_sweep_without_finetune_needs_a_model_named(logged_in_api):
    [error] = errors(logged_in_api, {"name": "qwen", "sweep": sweep_block()})

    assert error["loc"] == ["sweep", "model"]
    assert "name the `model`" in error["msg"]


@pytest.mark.parametrize(
    ("block", "name", "message"),
    [
        ("settings", "learning_rat", "not a SFTConfig setting"),
        ("settings", "output_dir", "is blocked"),
        ("lora", "task_type", "is blocked"),
        ("lora", "rank", "not a LoraConfig setting"),
    ],
)
def test_unknown_and_blocked_parameter_names_are_rejected(logged_in_api, block, name, message):
    request = sweeping(parameters={block: {name: {"values": [1, 2]}}})

    [error] = errors(logged_in_api, request)

    assert error["loc"] == ["sweep", "parameters", block, name]
    assert message in error["msg"]


def test_parameter_values_must_fit_the_setting(logged_in_api):
    [error] = errors(logged_in_api, sweeping(parameters={"lora": {"r": {"values": [8, "big"]}}}))

    assert error["loc"] == ["sweep", "parameters", "lora", "r"]
    assert "must match" in error["msg"]


def test_a_full_weight_sweep_varies_no_lora_setting(logged_in_api):
    request = sweep_only(method="full", lora=None, parameters={"lora": {"r": {"values": [8]}}})

    [error] = errors(logged_in_api, request)

    assert error["loc"] == ["sweep", "parameters", "lora"]
    assert "trains no Adapter" in error["msg"]


@pytest.mark.parametrize(
    "sweep",
    [
        {"objective": {"metric": "accuracy", "goal": "maximize"}},
        {"parameters": {}},
        {"parameters": {"settings": {"learning_rate": {"min": 1e-3, "max": 1e-5}}}},
        {"parameters": {"settings": {"learning_rate": {"min": 0, "max": 1, "scale": "log"}}}},
        {"parameters": {"settings": {"learning_rate": {"min": 0, "max": 1, "values": [1]}}}},
        {"sampler": "grid"},
        {"eval_split": 0.2, "eval_dataset": "dataset:chat"},
        {"trials": 0},
    ],
)
def test_invalid_searches_are_rejected(logged_in_api, sweep):
    assert validate(logged_in_api, sweep_only(**sweep)).status_code == 422


def test_an_unknown_objective_names_the_algorithms_metrics(logged_in_api):
    [error] = errors(logged_in_api, sweep_only(objective={"metric": "acc", "goal": "maximize"}))

    assert "optimizes one of eval_loss, eval_mean_token_accuracy, not acc" in error["msg"]


def test_the_eval_dataset_is_pinned_and_has_rows_the_algorithm_trains_on(logged_in_api):
    upload(logged_in_api, "pairs", jsonl({"prompt": "Hi", "chosen": "Hello!", "rejected": "Go."}))

    assert resolved(logged_in_api, sweep_only(eval_dataset="dataset:chat"))["sweep"][
        "eval_dataset"
    ] == ("dataset:chat@1")
    [error] = errors(logged_in_api, sweep_only(eval_dataset="dataset:pairs"))
    assert error["loc"] == ["sweep", "eval_dataset"]


def test_the_sweep_settings_are_checked_like_a_phases(logged_in_api):
    settings = {**sweep_block()["settings"], "learning_rat": 1}

    [error] = errors(logged_in_api, sweep_only(settings=settings))

    assert error["loc"] == ["sweep", "settings", "learning_rat"]


def test_params_from_needs_the_sweep_stage(logged_in_api):
    [error] = errors(logged_in_api, pipeline_request(phase={"params_from": "@sweep"}))

    assert error["loc"] == ["finetune", "phases", 0, "params_from"]
    assert "`sweep` isn't enabled" in error["msg"]


@pytest.mark.parametrize(
    ("sweep", "finetune", "mismatch"),
    [
        ({"method": "qlora"}, {}, "method"),
        ({"backend": "unsloth"}, {}, "backend"),
        ({"algorithm": "dpo", "dataset": "dataset:pairs"}, {}, "algorithm"),
    ],
)
def test_params_from_needs_the_sweeps_algorithm_method_and_backend(
    logged_in_api, sweep, finetune, mismatch
):
    upload(logged_in_api, "pairs", jsonl({"prompt": "Hi", "chosen": "Hello!", "rejected": "Go."}))

    [error] = errors(logged_in_api, sweeping(**sweep))

    assert error["loc"] == ["finetune", "phases", 0, "params_from"]
    assert f"`{mismatch}`" in error["msg"]


def test_params_from_cannot_set_lora_values_of_a_continued_adapter(logged_in_api):
    upload(logged_in_api, "pairs", jsonl({"prompt": "Hi", "chosen": "Hello!", "rejected": "Go."}))
    request = pipeline_request()
    request["finetune"]["phases"].append({**then("dpo", "dataset:pairs"), "params_from": "@sweep"})
    parameters = {"lora": {"r": {"values": [8]}}}
    sweep = {"algorithm": "dpo", "dataset": "dataset:pairs", "parameters": parameters}

    [error] = errors(logged_in_api, sweeping(request, **sweep))

    assert error["loc"] == ["finetune", "phases", 1, "params_from"]
    assert "continues the Adapter" in error["msg"]


def test_the_sweep_runs_before_finetune_which_gets_its_best_parameters(logged_in_api, cluster):
    request = sweeping()
    request["finetune"]["phases"].append(then("sft", "dataset:chat"))
    pipeline_id = submit(logged_in_api, request).json()["id"]

    sweep, sft, second = (tasks(cluster)[t] for t in ("sweep", "finetune", "finetune-2"))
    assert sweep["dependentTasks"] == ["fetch"]
    assert sft["dependentTasks"] == ["sweep"]
    assert parameters(sweep) == {
        "pipeline_id": str(pipeline_id),
        "request": parameters(sweep)["request"],
        "distilled_dataset": "",
    }
    assert parameters(sft)["swept_parameters"] == "sweep"
    assert parameters(second)["swept_parameters"] == ""


def test_the_sweep_trains_offline_on_its_backends_trainer_image(logged_in_api, cluster):
    submit(logged_in_api, sweep_only(backend="unsloth"))

    step = container(cluster, "sweep")
    assert step["image"] == "mlp-trainer-unsloth:test"
    assert step["command"] == ["mlp-stage", "sweep"]
    assert step["resources"]["accelerator"]["resourceCount"] == "1"
    assert {"name": "HF_HUB_OFFLINE", "value": "1"} in step["env"]
    assert {"name": "S3_BUCKET", "value": "platform"} in step["env"]


def test_a_sweep_alone_is_listed_as_a_stage_and_fetches_its_model(logged_in_api, cluster):
    submit(logged_in_api, sweep_only())

    assert set(tasks(cluster)) == {"fetch", "sweep"}
    assert parameters(tasks(cluster)["fetch"])["references"] == PINNED_BASE_MODEL
    [listed] = pipelines(logged_in_api)
    assert listed["stages"] == ["sweep"]


def test_the_sweep_step_stores_its_best_parameters_on_the_pipeline(logged_in_api):
    pipeline_id = submit(logged_in_api, sweep_only()).json()["id"]

    response = report(logged_in_api, pipeline_id)

    assert response.status_code == 201, response.text
    assert pipeline(logged_in_api, pipeline_id).json()["sweep"] == BEST
    assert pipelines(logged_in_api)[0]["sweep"] == BEST


def test_only_the_pipelines_own_step_reports_its_sweep(logged_in_api):
    first = submit(logged_in_api, sweep_only()).json()["id"]
    second = submit(logged_in_api, sweep_only()).json()["id"]
    token = logged_in_api.app.state.cluster.secrets[f"pipeline-{first}"]["step_token"]
    logged_in_api.cookies.clear()

    assert report(logged_in_api, second, token=token).status_code == 401


def test_a_pipeline_without_a_sweep_takes_no_report(logged_in_api):
    pipeline_id = submit(logged_in_api, pipeline_request()).json()["id"]
    token = issue_step_token(logged_in_api.app.state.jwt_secret, pipeline_id)

    assert report(logged_in_api, pipeline_id, token=token).status_code == 422


def test_a_finished_sweep_is_not_run_again_on_resume(logged_in_api, cluster):
    failed = submit(logged_in_api, sweeping()).json()["id"]
    report(logged_in_api, failed)
    end(logged_in_api, failed, "FAILED")

    assert resume(logged_in_api, failed).status_code == 202

    assert "sweep" not in tasks(cluster)
    swept = parameters(tasks(cluster)["finetune"])["swept_parameters"]
    assert json.loads(swept) == BEST["parameters"]


def test_a_failed_sweep_runs_again_on_resume(logged_in_api, cluster):
    failed = submit(logged_in_api, sweeping()).json()["id"]
    end(logged_in_api, failed, "FAILED")

    resume(logged_in_api, failed)

    assert parameters(tasks(cluster)["finetune"])["swept_parameters"] == "sweep"
