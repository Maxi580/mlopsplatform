import json

import pytest

from mlp_api.pipelines.cluster import KubeflowRun
from mlp_api.pipelines.hugging_face import HubModel
from mlp_api.pipelines.reconciler import reconcile_once, reconcile_pipelines
from mlp_core import api_paths, config

from .test_endpoint_stats import IDLE, SERVED
from .test_endpoints import register as register_tagged
from .test_endpoints import register_adapter
from .test_model_uploads import FULL_WEIGHTS
from .test_models import models, register

QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
TINY_QWEN = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
COMMIT = "c0ffee"
# The cases that run as Kubeflow nodes, without the serving cases' Endpoints.
WITHOUT_SERVING = {
    "finetune": {"phases": ["sft"], "methods": ["lora"], "backends": ["hf"]},
    "uploaded_model": True,
}
QUANTIZE_CASES = [
    "quantize-fp8-dynamic",
    "quantize-w4a16-gptq",
    "quantize-w4a16-awq",
    "quantize-w8a8-int8",
]
SPECULATE_CASES = [
    "speculate-eagle3",
    "speculate-dflash",
    "speculate-dspark",
    "speculate-peagle",
]
SERVING_CASES = [
    "serve-base-model",
    "serve-full-weights",
    "serve-adapter",
    "serve-merged",
    *(f"serve-{case}" for case in QUANTIZE_CASES),
    "serve-ngram",
    *(f"serve-{case}" for case in SPECULATE_CASES),
]
# Every Phase algorithm with every method of the `hf` backend, and those `unsloth` trains.
UNSLOTH = {"sft", "dpo", "kto", "grpo-lora", "grpo-qlora", "rloo-lora", "rloo-qlora"}
GRID_CASES = [
    f"{phase}-{method}-{backend}"
    for phase in ("sft", "dpo", "kto", "distillation", "grpo", "rloo")
    for method in ("lora", "qlora", "full")
    for backend in ("hf", "unsloth")
    if backend == "hf" or {phase, f"{phase}-{method}"} & UNSLOTH
]
WEIGHT_CASES = [
    "sft-rslora-hf",
    "sft-qlora-merged-hf",
    "sft-dora-merged-hf",
    "sft-lora-dpo-full-hf-finetune-sft",
    "sft-lora-dpo-full-hf",
    "sft-lora-merged-unsloth",
]
EVALUATE_CASES = [
    "evaluate-base-model",
    "evaluate-adapter",
    "evaluate-coding",
    "evaluate-evalscope",
    "evaluate-tool-calling",
    "evaluate-performance",
]


@pytest.fixture
def qwen_on_the_hub(logged_in_api, hugging_face):
    for repo in (QWEN, TINY_QWEN):
        hugging_face.models[repo] = HubModel(COMMIT, needs_remote_code=False, model_type="qwen2")
    # The assistant-only loss case checks that the Base Model's template marks assistant turns.
    hugging_face.files[QWEN] = {"chat_template.jinja": b"{% generation %}{% endgeneration %}"}
    # A download leaves its own bookkeeping next to the model files.
    hugging_face.files[TINY_QWEN] = {**FULL_WEIGHTS, ".cache/huggingface/download.lock": b""}


def start(api, selection=None):
    if selection is None:
        return api.post(api_paths.SMOKE_TEST_COMPLETE)
    return api.post(api_paths.SMOKE_TEST_CUSTOM, json=selection)


def nodes(cluster) -> list[dict]:
    """The submitted run's tasks in the order they run."""
    tasks = cluster.submitted["pipeline_spec"]["pipeline_spec"]["root"]["dag"]["tasks"]
    ordered, previous = [], None
    while len(ordered) < len(tasks):
        [task] = [t for t in tasks.values() if t.get("dependentTasks", [None])[0] == previous]
        ordered.append(task)
        previous = next(key for key, t in tasks.items() if t is task)
    return ordered


def node(cluster, case: str) -> dict:
    [found] = [task for task in nodes(cluster) if task["taskInfo"]["name"] == case]
    return found


def case_names(cluster) -> list[str]:
    return [task["taskInfo"]["name"] for task in nodes(cluster)]


def smoke_test(api) -> dict:
    [found] = [p for p in api.get(api_paths.PIPELINES).json() if p["cases"] is not None]
    return found


def test_the_complete_smoke_test_runs_fetch_then_every_finetune_case(
    logged_in_api, qwen_on_the_hub, cluster
):
    response = start(logged_in_api)

    assert response.status_code == 202, response.text
    started = response.json()
    assert started["name"].startswith("smoketest-")
    assert started["kubeflow_run_url"] == "/pipeline/#/runs/details/run-1"
    assert case_names(cluster) == [
        "fetch",
        "sandbox",
        *GRID_CASES,
        "sft-assistant-only-hf",
        "sft-assistant-only-unsloth",
        "uploaded-model",
        "distill-tools-distill",
        "distill-tools",
        "sft-sweep-sweep",
        "sft-sweep",
        "sft-dpo-chain-finetune-sft",
        "sft-dpo-chain",
        *WEIGHT_CASES,
        "resume-interrupted",
        "resume",
        *QUANTIZE_CASES,
        "quantize-adapter",
        *SPECULATE_CASES,
        "evaluate-base-model",
        "evaluate-adapter",
        "evaluate-coding",
        "evaluate-evalscope",
        "evaluate-tool-calling",
        "evaluate-performance",
    ]
    assert cluster.submitted["display_name"] == started["name"]


def test_every_case_runs_even_after_the_one_before_it_failed(
    logged_in_api, qwen_on_the_hub, cluster
):
    start(logged_in_api)

    # Only a step that follows its case's earlier step waits for that to pass.
    first, *later = nodes(cluster)
    assert "triggerPolicy" not in first
    waiting = ("distill-tools", "sft-sweep", "sft-dpo-chain", "sft-lora-dpo-full-hf", "resume")
    for waits in waiting:
        assert "triggerPolicy" not in node(cluster, waits)
        later.remove(node(cluster, waits))
    assert later
    assert all(t["triggerPolicy"]["strategy"] == "ALL_UPSTREAM_TASKS_COMPLETED" for t in later)


@pytest.mark.parametrize(
    ("selection", "cases"),
    [
        ({}, ["fetch"]),
        ({"uploaded_model": True}, ["fetch", "uploaded-model"]),
        ({"sandbox": True}, ["fetch", "sandbox"]),
        ({"quantize": True}, ["fetch", *QUANTIZE_CASES]),
        ({"speculate": True}, ["fetch", *SPECULATE_CASES]),
        (
            {"quantize": True, "finetune": {"phases": ["sft"], "methods": ["lora"]}},
            ["fetch", "sft-lora-hf", "sft-lora-unsloth", "sft-assistant-only-hf"]
            + ["sft-assistant-only-unsloth", *QUANTIZE_CASES, "quantize-adapter"],
        ),
        (
            {"evaluate": True},
            ["fetch", "evaluate-base-model", "evaluate-coding", "evaluate-evalscope"]
            + ["evaluate-tool-calling", "evaluate-performance"],
        ),
        (
            {"finetune": {"phases": ["sft"], "backends": ["hf"]}},
            ["fetch", "sft-lora-hf", "sft-qlora-hf", "sft-full-hf", "sft-assistant-only-hf"],
        ),
        (
            {"finetune": {"phases": ["sft"], "methods": ["full"]}},
            ["fetch", "sft-full-hf", "sft-full-unsloth"],
        ),
        (
            {"finetune": {"phases": ["grpo", "distillation"], "backends": ["unsloth"]}},
            ["fetch", "grpo-lora-unsloth", "grpo-qlora-unsloth"],
        ),
        ({"finetune": {"phases": ["sft"], "methods": [], "backends": ["hf"]}}, ["fetch"]),
    ],
)
def test_a_custom_smoke_test_runs_only_the_selected_cases(
    logged_in_api, qwen_on_the_hub, cluster, selection, cases
):
    response = start(logged_in_api, selection)

    assert response.status_code == 202, response.text
    assert case_names(cluster) == cases
    assert list(smoke_test(logged_in_api)["cases"]) == cases


@pytest.mark.parametrize(
    "selection",
    [{"finetune": {"phases": ["ppo"]}}, {"finetune": {"backends": ["axolotl"]}}, {"quiz": True}],
)
def test_a_selection_of_unknown_cases_is_rejected(
    logged_in_api, qwen_on_the_hub, cluster, selection
):
    response = start(logged_in_api, selection)

    assert response.status_code == 422
    assert cluster.runs == {}


def test_a_finetune_case_trains_the_pinned_base_model_on_its_uploaded_bundled_dataset(
    logged_in_api, qwen_on_the_hub, cluster
):
    name = start(logged_in_api).json()["name"]

    finetune = node(cluster, "sft-lora-hf")
    parameters = finetune["inputs"]["parameters"]
    request = json.loads(parameters["request"]["runtimeValue"]["constant"])
    assert request["name"] == f"{name}-sft-lora-hf"
    assert request["finetune"]["base_model"] == f"hf:{QWEN}@{COMMIT}"
    [phase] = request["finetune"]["phases"]
    assert phase["dataset"] == f"dataset:{name}-sft@1"
    datasets = logged_in_api.get(api_paths.DATASETS).json()
    bundled = ("calibration", "distill", "distillation", "dpo", "grpo", "kto", "rloo", "sft")
    assert [d["name"] for d in datasets] == [f"{name}-{dataset}" for dataset in bundled]


def test_a_distillation_case_learns_from_the_base_model_as_teacher_on_bundled_prompts(
    logged_in_api, qwen_on_the_hub, cluster
):
    name = start(logged_in_api, {"finetune": {"phases": ["distillation"], "methods": ["lora"]}})
    name = name.json()["name"]

    [phase] = phases(cluster, "distillation-lora-hf")
    assert phase["teacher"] == f"hf:{QWEN}@{COMMIT}"
    assert phase["dataset"] == f"dataset:{name}-distillation@1"
    assert phase["settings"]["max_completion_length"] == 256
    assert "max_length" not in phase["settings"]


@pytest.mark.parametrize("algorithm", ["grpo", "rloo"])
def test_an_rl_case_learns_from_two_weighted_rewards_on_bundled_prompts(
    logged_in_api, qwen_on_the_hub, cluster, algorithm
):
    name = start(logged_in_api, {"finetune": {"phases": [algorithm], "methods": ["lora"]}})
    name = name.json()["name"]

    [phase] = phases(cluster, f"{algorithm}-lora-hf")
    assert phase["dataset"] == f"dataset:{name}-{algorithm}@1"
    assert [reward["weight"] for reward in phase["rewards"].values()] == [0.8, 0.2]
    # Each step's batch holds whole groups of completions to one prompt.
    assert phase["settings"]["num_generations"] == 2
    assert phase["settings"]["max_completion_length"] == 256


def test_the_uploaded_model_case_finetunes_a_tiny_model_uploaded_the_normal_way(
    logged_in_api, qwen_on_the_hub, cluster
):
    name = start(logged_in_api).json()["name"]

    uploaded_model = node(cluster, "uploaded-model")
    parameters = uploaded_model["inputs"]["parameters"]
    request = json.loads(parameters["request"]["runtimeValue"]["constant"])
    assert request["name"] == f"{name}-uploaded-model"
    assert request["finetune"]["from"] == f"model:{name}-uploaded@1"
    [version] = models(logged_in_api)[f"{name}-uploaded"]
    assert (version["tags"]["source"], version["tags"]["weights"]) == ("uploaded", "full")
    assert version["size_bytes"] == sum(len(content) for content in FULL_WEIGHTS.values())


def test_the_distill_case_distills_with_the_base_model_and_a_tool_then_trains_on_it(
    logged_in_api, qwen_on_the_hub, cluster
):
    started = start(logged_in_api, {"distill": True}).json()
    name = started["name"]

    assert case_names(cluster) == ["fetch", "distill-tools-distill", "distill-tools"]
    assert list(smoke_test(logged_in_api)["cases"]) == ["fetch", "distill-tools"]
    distill, finetune = node(cluster, "distill-tools-distill"), node(cluster, "distill-tools")
    request = json.loads(distill["inputs"]["parameters"]["request"]["runtimeValue"]["constant"])
    assert request["distill"]["teacher"] == f"hf:{QWEN}@{COMMIT}"
    assert request["distill"]["dataset"] == f"dataset:{name}-distill@1"
    assert request["distill"]["tools"][0]["function"]["name"] == "get_weather"
    assert request["distill"]["serving"]["tool_parser"] == "hermes"
    assert request["finetune"]["phases"][0]["dataset"] == "@distill"
    handoff = finetune["inputs"]["parameters"]["distilled_dataset"]["taskOutputParameter"]
    assert handoff["outputParameterKey"] == "dataset"
    assert set(cluster.secrets[f"pipeline-{started['id']}"]) == {"step_token"}


def test_the_sweep_case_sweeps_two_trials_then_trains_with_the_best_parameters(
    logged_in_api, qwen_on_the_hub, cluster
):
    started = start(logged_in_api, {"sweep": True}).json()
    name = started["name"]

    assert case_names(cluster) == ["fetch", "sft-sweep-sweep", "sft-sweep"]
    assert list(smoke_test(logged_in_api)["cases"]) == ["fetch", "sft-sweep"]
    sweep, finetune = node(cluster, "sft-sweep-sweep"), node(cluster, "sft-sweep")
    request = json.loads(sweep["inputs"]["parameters"]["request"]["runtimeValue"]["constant"])
    assert request["sweep"]["model"] == f"hf:{QWEN}@{COMMIT}"
    assert request["sweep"]["dataset"] == f"dataset:{name}-sft@1"
    assert request["sweep"]["trials"] == 2
    assert request["finetune"]["phases"][0]["params_from"] == "@sweep"
    handoff = finetune["inputs"]["parameters"]["swept_parameters"]["taskOutputParameter"]
    assert handoff["outputParameterKey"] == "best_parameters"
    assert set(cluster.secrets[f"pipeline-{started['id']}"]) == {"step_token"}


def test_the_sweep_cases_step_reports_its_best_parameters(logged_in_api, qwen_on_the_hub, cluster):
    started = start(logged_in_api, {"sweep": True}).json()
    token = cluster.secrets[f"pipeline-{started['id']}"]["step_token"]
    best = {"parameters": {"settings": {"learning_rate": 1e-4}}, "objective": 1.5}

    response = logged_in_api.post(
        api_paths.SWEEP_PIPELINE.format(id=started["id"]),
        json=best,
        headers={"authorization": f"Bearer {token}"},
    )

    assert response.status_code == 201, response.text
    assert smoke_test(logged_in_api)["sweep"] == best


def test_the_chain_case_trains_sft_then_dpo_on_the_sft_phases_model_version(
    logged_in_api, qwen_on_the_hub, cluster
):
    name = start(logged_in_api, {"chain": True}).json()["name"]

    assert case_names(cluster) == ["fetch", "sft-dpo-chain-finetune-sft", "sft-dpo-chain"]
    assert list(smoke_test(logged_in_api)["cases"]) == ["fetch", "sft-dpo-chain"]
    sft, dpo = node(cluster, "sft-dpo-chain-finetune-sft"), node(cluster, "sft-dpo-chain")
    request = json.loads(sft["inputs"]["parameters"]["request"]["runtimeValue"]["constant"])
    first, second = request["finetune"]["phases"]
    assert (first["algorithm"], first["dataset"]) == ("sft", f"dataset:{name}-sft@1")
    assert (second["algorithm"], second["dataset"]) == ("dpo", f"dataset:{name}-dpo@1")
    assert first["lora"]["r"] == 8 and second["lora"] is None
    handoff = dpo["inputs"]["parameters"]["previous_model_version"]["taskOutputParameter"]
    assert handoff["outputParameterKey"] == "model_version"


def test_the_resume_case_stops_a_phase_after_its_first_checkpoint_then_continues_it(
    logged_in_api, qwen_on_the_hub, cluster
):
    smoke_test_id = start(logged_in_api, {"resume": True}).json()["id"]

    assert case_names(cluster) == ["fetch", "resume-interrupted", "resume"]
    assert list(smoke_test(logged_in_api)["cases"]) == ["fetch", "resume"]
    interrupted, resumed = (
        {
            name: p["runtimeValue"]["constant"]
            for name, p in node(cluster, case)["inputs"]["parameters"].items()
        }
        for case in ("resume-interrupted", "resume")
    )
    assert interrupted["checkpoint_minutes"] == "0"
    assert interrupted["stop_after_checkpoint"] == "true"
    assert interrupted["resume_checkpoint"] == ""
    assert resumed["checkpoint_minutes"] == "30"
    assert resumed["stop_after_checkpoint"] == ""
    assert resumed["resume_checkpoint"] == f"checkpoints/{smoke_test_id}/0/"


def phases(cluster, case: str) -> list[dict]:
    parameters = node(cluster, case)["inputs"]["parameters"]
    return json.loads(parameters["request"]["runtimeValue"]["constant"])["finetune"]["phases"]


def test_a_full_case_trains_every_weight_the_others_an_adapter_one_assistant_only(
    logged_in_api, qwen_on_the_hub, cluster
):
    start(logged_in_api, {"finetune": {"phases": ["sft"], "backends": ["hf"]}})

    trained = {case: phases(cluster, case)[0] for case in case_names(cluster)[1:]}
    assert {case: phase["method"] for case, phase in trained.items()} == {
        "sft-lora-hf": "lora",
        "sft-qlora-hf": "qlora",
        "sft-full-hf": "full",
        "sft-assistant-only-hf": "lora",
    }
    assert trained["sft-full-hf"]["lora"] is None
    assert trained["sft-qlora-hf"]["lora"]["r"] == 8
    assert trained["sft-assistant-only-hf"]["settings"]["assistant_only_loss"] is True


def test_the_weight_cases_try_rslora_merged_outputs_and_a_full_phase_after_an_adapter(
    logged_in_api, qwen_on_the_hub, cluster
):
    start(logged_in_api, {"weights": True})

    assert case_names(cluster) == ["fetch", *WEIGHT_CASES]
    rslora, qlora, dora = (phases(cluster, case)[0] for case in WEIGHT_CASES[:3])
    assert (rslora["output"], rslora["lora"]["use_rslora"]) == ("adapter", True)
    assert (qlora["method"], qlora["output"]) == ("qlora", "merged")
    assert (dora["output"], dora["lora"]["use_dora"]) == ("merged", True)
    sft, dpo = phases(cluster, "sft-lora-dpo-full-hf")
    assert (sft["method"], dpo["method"], dpo["lora"]) == ("lora", "full", None)


def test_the_merged_model_is_served_once_its_weight_case_registered_it(
    logged_in_api, qwen_on_the_hub, cluster, model_registry, object_store
):
    name = start(logged_in_api, {"weights": True, "serving": True}).json()["name"]
    register(model_registry, object_store, f"{name}-sft-qlora-merged-hf", 1, weights="full")
    cluster.runs["run-1"] = KubeflowRun("RUNNING", None, {"sft-qlora-merged-hf": "SUCCEEDED"})

    reconcile(logged_in_api)

    merged = running_endpoints(logged_in_api)[f"{name}-serve-merged"]
    assert merged == f"model:{name}-sft-qlora-merged-hf@1"


def test_the_distill_cases_step_registers_its_dataset_under_the_cases_name(
    logged_in_api, qwen_on_the_hub, cluster
):
    started = start(logged_in_api, {"distill": True}).json()
    token = cluster.secrets[f"pipeline-{started['id']}"]["step_token"]
    row = {"prompt": "Hi", "completion": "Hello"}

    response = logged_in_api.post(
        api_paths.DISTILL_PIPELINE.format(id=started["id"]),
        content=json.dumps(row) + "\n",
        headers={"authorization": f"Bearer {token}"},
    )

    assert response.status_code == 201, response.text
    assert response.json() == {"dataset": f"dataset:{started['name']}-distill-tools@1"}


def test_the_sandbox_case_runs_snippets_in_the_sandbox_from_a_pipeline_step(
    logged_in_api, qwen_on_the_hub, cluster
):
    start(logged_in_api, {"sandbox": True})

    _, sandbox = nodes(cluster)
    assert sandbox["triggerPolicy"]["strategy"] == "ALL_UPSTREAM_TASKS_COMPLETED"
    pipeline = cluster.submitted["pipeline_spec"]["pipeline_spec"]
    container = pipeline["deploymentSpec"]["executors"]["exec-sandbox"]["container"]
    assert container["command"] == ["mlp-stage", "check-sandbox"]
    assert {"name": "SANDBOX_URL", "value": "http://sandbox.mlp.test:8090"} in container["env"]
    memory = sandbox["inputs"]["parameters"]["sandbox_memory_mb"]["runtimeValue"]["constant"]
    assert memory == "1024"


def test_the_evaluate_cases_run_a_few_samples_of_their_benchmark_on_the_base_model_or_adapter(
    logged_in_api, qwen_on_the_hub, cluster
):
    name = start(logged_in_api).json()["name"]

    fetch = node(cluster, "fetch")["inputs"]["parameters"]["references"]["runtimeValue"]
    assert fetch["constant"] == (
        f"hf:{QWEN}@{COMMIT},lm_eval:truthfulqa_mc2,lm_eval:humaneval,evalscope:mbpp_plus,"
        "bfcl:simple_python"
    )
    evaluated = {}
    for case in [case for case in EVALUATE_CASES if case != "evaluate-performance"]:
        parameters = node(cluster, case)["inputs"]["parameters"]
        request = json.loads(parameters["request"]["runtimeValue"]["constant"])
        assert request["evaluate"]["limit"] == 5
        evaluated[case] = (request["evaluate"]["model"], *request["evaluate"]["benchmarks"])
    base_model = f"hf:{QWEN}@{COMMIT}"
    assert evaluated == {
        "evaluate-base-model": (base_model, "lm_eval:truthfulqa_mc2"),
        "evaluate-adapter": (f"model:{name}-sft-lora-hf@1", "lm_eval:truthfulqa_mc2"),
        "evaluate-coding": (base_model, "lm_eval:humaneval"),
        "evaluate-evalscope": (base_model, "evalscope:mbpp_plus"),
        "evaluate-tool-calling": (base_model, "bfcl:simple_python"),
    }


def test_the_quantize_cases_quantize_the_base_model_calibrating_on_bundled_rows_and_an_adapter(
    logged_in_api, qwen_on_the_hub, cluster
):
    name = start(logged_in_api).json()["name"]

    quantized = {}
    for case in [*QUANTIZE_CASES, "quantize-adapter"]:
        parameters = node(cluster, case)["inputs"]["parameters"]
        request = json.loads(parameters["request"]["runtimeValue"]["constant"])
        assert request["name"] == f"{name}-{case}"
        quantize = request["quantize"]
        quantized[case] = (quantize["model"], quantize["scheme"], quantize["calibration"])
    base_model = f"hf:{QWEN}@{COMMIT}"
    calibration = {"dataset": f"dataset:{name}-calibration@1", "samples": 16, "max_length": 256}
    assert quantized == {
        "quantize-fp8-dynamic": (base_model, "fp8-dynamic", None),
        "quantize-w4a16-gptq": (base_model, "w4a16-gptq", calibration),
        "quantize-w4a16-awq": (base_model, "w4a16-awq", calibration),
        "quantize-w8a8-int8": (base_model, "w8a8-int8", calibration),
        "quantize-adapter": (f"model:{name}-sft-lora-hf@1", "fp8-dynamic", None),
    }


def test_each_quantized_model_is_served_once_its_quantize_case_registered_it(
    logged_in_api, qwen_on_the_hub, cluster, model_registry, object_store
):
    name = start(logged_in_api, {"quantize": True, "serving": True}).json()["name"]
    quantized = f"{name}-quantize-w4a16-awq"
    register(model_registry, object_store, quantized, 1, weights="full")
    steps = {"quantize-w4a16-awq": "SUCCEEDED", "quantize-w8a8-int8": "FAILED"}
    cluster.runs["run-1"] = KubeflowRun("RUNNING", None, steps)

    reconcile(logged_in_api)

    assert running_endpoints(logged_in_api)[f"{name}-serve-quantize-w4a16-awq"] == (
        f"model:{quantized}@1"
    )
    cases = smoke_test(logged_in_api)["cases"]
    assert (cases["serve-quantize-w8a8-int8"], cases["serve-quantize-fp8-dynamic"]) == (
        "failed",
        "pending",
    )


def test_the_tool_calling_case_serves_the_base_model_with_its_tool_parser(
    logged_in_api, qwen_on_the_hub, cluster
):
    start(logged_in_api)

    parameters = node(cluster, "evaluate-tool-calling")["inputs"]["parameters"]
    evaluate = json.loads(parameters["request"]["runtimeValue"]["constant"])["evaluate"]
    assert evaluate["serving"]["tool_parser"] == "hermes"


def test_the_performance_case_runs_one_short_guidellm_run_on_the_base_model(
    logged_in_api, qwen_on_the_hub, cluster
):
    start(logged_in_api)

    parameters = node(cluster, "evaluate-performance")["inputs"]["parameters"]
    evaluate = json.loads(parameters["request"]["runtimeValue"]["constant"])["evaluate"]
    assert evaluate["model"] == f"hf:{QWEN}@{COMMIT}"
    assert evaluate["benchmarks"] == []
    assert evaluate["performance"]["requests"] <= 10


def test_only_one_smoke_test_runs_at_a_time(logged_in_api, qwen_on_the_hub, cluster):
    start(logged_in_api)

    response = start(logged_in_api, {})

    assert response.status_code == 409
    assert len(cluster.runs) == 1


def reconcile(api):
    reconcile_once(api.app.state)


def test_each_case_reports_its_own_result(logged_in_api, qwen_on_the_hub, cluster):
    start(logged_in_api, WITHOUT_SERVING)
    cluster.runs["run-1"] = KubeflowRun("RUNNING", None, {"fetch": "FAILED"})

    reconcile(logged_in_api)

    assert smoke_test(logged_in_api)["cases"] == {
        "fetch": "failed",
        "sft-lora-hf": "pending",
        "sft-assistant-only-hf": "pending",
        "uploaded-model": "pending",
    }


def test_a_failed_case_does_not_stop_the_others(logged_in_api, qwen_on_the_hub, cluster):
    start(logged_in_api, WITHOUT_SERVING)
    cluster.runs["run-1"] = KubeflowRun(
        "FAILED", None, {"fetch": "FAILED", "sft-lora-hf": "SUCCEEDED"}
    )

    reconcile(logged_in_api)

    found = smoke_test(logged_in_api)
    assert found["status"] == "failed"
    assert found["cases"] == {
        "fetch": "failed",
        "sft-lora-hf": "passed",
        "sft-assistant-only-hf": "failed",
        "uploaded-model": "failed",
    }


def test_a_case_that_never_ran_in_a_finished_run_failed(logged_in_api, qwen_on_the_hub, cluster):
    start(logged_in_api, WITHOUT_SERVING)
    cluster.runs["run-1"] = KubeflowRun("FAILED", None, {"fetch": "SUCCEEDED"})

    reconcile(logged_in_api)

    assert smoke_test(logged_in_api)["cases"] == {
        "fetch": "passed",
        "sft-lora-hf": "failed",
        "sft-assistant-only-hf": "failed",
        "uploaded-model": "failed",
    }


def test_afterwards_only_its_kubeflow_run_remains(
    logged_in_api, qwen_on_the_hub, cluster, model_registry, object_store
):
    name = start(logged_in_api, WITHOUT_SERVING).json()["name"]
    smoke_test_id = smoke_test(logged_in_api)["id"]
    register(model_registry, object_store, f"{name}-sft-lora-hf", 1, pipeline_id=smoke_test_id)
    register(model_registry, object_store, "qwen-sft", 1)
    object_store.objects[f"checkpoints/{smoke_test_id}/0/checkpoint-1/trainer_state.json"] = b"{}"
    cluster.runs["run-1"] = KubeflowRun("SUCCEEDED", None, {"fetch": "SUCCEEDED"})

    reconcile(logged_in_api)

    assert logged_in_api.get(api_paths.DATASETS).json() == []
    assert [v.name for v in model_registry.versions] == ["qwen-sft"]
    assert model_registry.deleted_models == [f"{name}-sft-lora-hf", f"{name}-uploaded"]
    assert object_store.objects == {}
    assert list(object_store.buckets["mlflow"]) == [
        "1/run-qwen-sft-1/artifacts/model/adapter_model.safetensors"
    ]
    assert cluster.secrets == {}
    assert (list(cluster.runs), cluster.terminated) == (["run-1"], [])


def test_a_finished_smoke_test_can_be_deleted_once_it_is_cleaned_up(
    logged_in_api, qwen_on_the_hub, cluster
):
    smoke_test_id = start(logged_in_api, WITHOUT_SERVING).json()["id"]
    cluster.runs["run-1"] = KubeflowRun("SUCCEEDED", None, {"fetch": "SUCCEEDED"})
    reconcile_pipelines(logged_in_api.app.state.engine, cluster)
    path = api_paths.PIPELINE.format(id=smoke_test_id)

    refused = logged_in_api.delete(path)
    reconcile(logged_in_api)
    deleted = logged_in_api.delete(path)

    assert refused.status_code == 409
    assert "cleaning up" in refused.json()["detail"]
    assert deleted.status_code == 204, deleted.text


def test_its_datasets_stay_while_it_runs(logged_in_api, qwen_on_the_hub, cluster):
    start(logged_in_api)
    cluster.runs["run-1"] = KubeflowRun("RUNNING", None)

    reconcile(logged_in_api)

    assert len(logged_in_api.get(api_paths.DATASETS).json()) == 8


def test_a_smoke_test_that_could_not_start_is_failed_and_cleaned_up(
    logged_in_api, hugging_face, cluster
):
    response = start(logged_in_api)

    assert response.status_code == 502
    assert QWEN in response.json()["detail"]
    assert smoke_test(logged_in_api)["status"] == "failed"
    reconcile(logged_in_api)
    assert logged_in_api.get(api_paths.DATASETS).json() == []


def test_smoke_tests_require_login(api, cluster):
    assert start(api).status_code == 401


def test_cancelling_a_smoke_test_fails_its_unfinished_cases(
    logged_in_api, qwen_on_the_hub, cluster
):
    smoke_test_id = start(logged_in_api, WITHOUT_SERVING).json()["id"]
    cluster.runs["run-1"] = KubeflowRun("RUNNING", None, {"fetch": "SUCCEEDED"})
    reconcile(logged_in_api)

    logged_in_api.post(api_paths.CANCEL_PIPELINE.format(id=smoke_test_id))

    assert smoke_test(logged_in_api)["cases"] == {
        "fetch": "passed",
        "sft-lora-hf": "failed",
        "sft-assistant-only-hf": "failed",
        "uploaded-model": "failed",
    }


def running_endpoints(api) -> dict[str, str]:
    """Each Endpoint that isn't stopped, by name, with the model it serves."""
    listed = api.get(api_paths.ENDPOINTS).json()
    return {e["name"]: e["model"] for e in listed if e["status"] != "stopped"}


def test_the_complete_smoke_test_serves_the_base_model_and_the_uploaded_full_weights(
    logged_in_api, qwen_on_the_hub, cluster
):
    name = start(logged_in_api).json()["name"]

    reconcile(logged_in_api)

    assert list(smoke_test(logged_in_api)["cases"]) == [
        "fetch",
        "sandbox",
        *GRID_CASES,
        "sft-assistant-only-hf",
        "sft-assistant-only-unsloth",
        "uploaded-model",
        "distill-tools",
        "sft-sweep",
        "sft-dpo-chain",
        *[case for case in WEIGHT_CASES if not case.endswith("-finetune-sft")],
        "resume",
        *QUANTIZE_CASES,
        "quantize-adapter",
        *SPECULATE_CASES,
        *EVALUATE_CASES,
        *SERVING_CASES,
    ]
    assert running_endpoints(logged_in_api) == {
        f"{name}-serve-base-model": f"hf:{QWEN}@{COMMIT}",
        f"{name}-serve-full-weights": f"model:{name}-uploaded@1",
        f"{name}-serve-ngram": f"hf:{QWEN}@{COMMIT}",
    }


def test_a_custom_smoke_test_can_serve_without_finetuning(logged_in_api, qwen_on_the_hub, cluster):
    name = start(logged_in_api, {"serving": True}).json()["name"]

    reconcile(logged_in_api)

    assert list(smoke_test(logged_in_api)["cases"]) == ["fetch", *SERVING_CASES[:2], "serve-ngram"]
    assert list(running_endpoints(logged_in_api)) == [
        f"{name}-serve-ngram",
        f"{name}-serve-full-weights",
        f"{name}-serve-base-model",
    ]


def test_the_adapter_is_served_once_its_finetune_case_registered_it(
    logged_in_api, qwen_on_the_hub, cluster, model_registry, object_store
):
    name = start(logged_in_api).json()["name"]
    register_adapter(model_registry, object_store, f"{name}-sft-lora-hf", f"hf:{QWEN}@{COMMIT}")
    steps = {"fetch": "SUCCEEDED", "sft-lora-hf": "SUCCEEDED"}
    cluster.runs["run-1"] = KubeflowRun("RUNNING", None, steps)

    reconcile(logged_in_api)

    adapter = running_endpoints(logged_in_api)[f"{name}-serve-adapter"]
    assert adapter == f"model:{name}-sft-lora-hf@1"


def test_the_adapter_case_fails_with_its_finetune_case(logged_in_api, qwen_on_the_hub, cluster):
    name = start(logged_in_api).json()["name"]
    cluster.runs["run-1"] = KubeflowRun("RUNNING", None, {"sft-lora-hf": "FAILED"})

    reconcile(logged_in_api)

    assert smoke_test(logged_in_api)["cases"]["serve-adapter"] == "failed"
    assert f"{name}-serve-adapter" not in running_endpoints(logged_in_api)


def test_a_serving_case_passes_once_its_endpoint_answered_a_chat_request_and_then_stops_it(
    logged_in_api, qwen_on_the_hub, cluster
):
    name = start(logged_in_api).json()["name"]
    reconcile(logged_in_api)
    cluster.endpoint_states[f"{name}-serve-base-model"] = "running"
    cluster.metrics[f"{name}-serve-base-model"] = SERVED
    cluster.endpoint_states[f"{name}-serve-full-weights"] = "failed"
    cluster.endpoint_states[f"{name}-serve-ngram"] = "failed"

    reconcile(logged_in_api)

    cases = smoke_test(logged_in_api)["cases"]
    assert (cases["serve-base-model"], cases["serve-full-weights"]) == ("passed", "failed")
    assert cluster.chat_requests == [(f"{name}-serve-base-model", config.SMOKE_TEST_CHAT_REQUEST)]
    assert cluster.endpoints == {}


@pytest.mark.parametrize(
    "metrics",
    [
        # vLLM is ready but doesn't answer.
        None,
        # Its stats count no finished request.
        IDLE,
    ],
    ids=["unreachable", "idle"],
)
def test_a_serving_case_fails_unless_its_stats_count_the_chat_request(
    logged_in_api, qwen_on_the_hub, cluster, metrics
):
    name = start(logged_in_api, {"serving": True}).json()["name"]
    reconcile(logged_in_api)
    cluster.endpoint_states[f"{name}-serve-base-model"] = "running"
    if metrics:
        cluster.metrics[f"{name}-serve-base-model"] = metrics

    reconcile(logged_in_api)

    assert smoke_test(logged_in_api)["cases"]["serve-base-model"] == "failed"
    assert f"{name}-serve-base-model" not in running_endpoints(logged_in_api)


def test_a_smoke_test_finishes_once_its_serving_cases_did(
    logged_in_api, qwen_on_the_hub, cluster, model_registry, object_store
):
    name = start(logged_in_api).json()["name"]
    register_adapter(model_registry, object_store, f"{name}-sft-lora-hf", f"hf:{QWEN}@{COMMIT}")
    for full_weights in ("sft-qlora-merged-hf", *QUANTIZE_CASES):
        register(model_registry, object_store, f"{name}-{full_weights}", 1, weights="full")
    for case in SPECULATE_CASES:
        tags = {"speculator": case.removeprefix("speculate-"), "verifier": f"hf:{QWEN}@{COMMIT}"}
        register_tagged(model_registry, object_store, f"{name}-{case}-speculator", tags)
    steps = dict.fromkeys(case_names(cluster), "SUCCEEDED")
    cluster.runs["run-1"] = KubeflowRun("SUCCEEDED", None, steps)

    reconcile(logged_in_api)
    waiting = smoke_test(logged_in_api)["status"]
    for case in SERVING_CASES:
        cluster.endpoint_states[f"{name}-{case}"] = "running"
        cluster.metrics[f"{name}-{case}"] = SERVED
    reconcile(logged_in_api)
    reconcile(logged_in_api)

    assert waiting == "running"
    found = smoke_test(logged_in_api)
    assert found["status"] == "succeeded"
    assert set(found["cases"].values()) == {"passed"}
    assert model_registry.versions == []


def test_a_speculate_case_trains_its_type_for_the_base_model_on_the_bundled_conversations(
    logged_in_api, qwen_on_the_hub, cluster
):
    name = start(logged_in_api, {"speculate": True}).json()["name"]

    parameters = node(cluster, "speculate-peagle")["inputs"]["parameters"]
    request = json.loads(parameters["request"]["runtimeValue"]["constant"])
    assert request["speculate"] == {
        "speculator": "peagle",
        "model": f"hf:{QWEN}@{COMMIT}",
        "dataset": f"dataset:{name}-sft@1",
        "settings": {**config.SMOKE_TEST_SPECULATE_SETTINGS, "learning_rate": 1e-4},
    }


def test_a_speculator_is_served_once_its_speculate_case_registered_it(
    logged_in_api, qwen_on_the_hub, cluster, model_registry, object_store
):
    name = start(logged_in_api, {"speculate": True, "serving": True}).json()["name"]
    reconcile(logged_in_api)
    tags = {"speculator": "dflash", "verifier": f"hf:{QWEN}@{COMMIT}"}
    register_tagged(model_registry, object_store, f"{name}-speculate-dflash-speculator", tags)
    cluster.runs["run-1"] = KubeflowRun("RUNNING", None, {"speculate-dflash": "SUCCEEDED"})

    reconcile(logged_in_api)

    served = {name: endpoint["spec"] for name, endpoint in endpoint_specs(logged_in_api).items()}
    assert served[f"{name}-serve-speculate-dflash"]["speculative"] == {
        "method": "dflash",
        "model": f"model:{name}-speculate-dflash-speculator@1",
    }
    assert served[f"{name}-serve-ngram"]["speculative"] == config.SMOKE_TEST_NGRAM
    assert f"{name}-serve-speculate-eagle3" not in served


def endpoint_specs(api) -> dict[str, dict]:
    return {e["name"]: e for e in api.get(api_paths.ENDPOINTS).json() if e["status"] != "stopped"}


def test_a_cancelled_smoke_test_stops_its_endpoints(logged_in_api, qwen_on_the_hub, cluster):
    smoke_test_id = start(logged_in_api).json()["id"]
    reconcile(logged_in_api)

    logged_in_api.post(api_paths.CANCEL_PIPELINE.format(id=smoke_test_id))
    reconcile(logged_in_api)

    assert cluster.endpoints == {}
    assert running_endpoints(logged_in_api) == {}


def test_a_smoke_test_without_distill_or_sweep_has_no_secret(
    logged_in_api, qwen_on_the_hub, cluster
):
    start(logged_in_api, WITHOUT_SERVING)

    assert cluster.secrets == {}
