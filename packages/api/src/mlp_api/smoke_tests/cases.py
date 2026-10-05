from typing import Literal

from mlp_api.pipelines.cluster import KubeflowRun
from mlp_core import config
from mlp_core.pipeline_request.references import model_reference
from mlp_core.pipeline_request.schema import METHODS, Strict


class FinetuneCases(Strict):
    """Every combination of these that a backend supports is one case; unnamed means all."""

    phases: list[Literal[tuple(config.ALGORITHMS)]] = list(config.ALGORITHMS)
    methods: list[Literal[METHODS]] = list(METHODS)
    backends: list[Literal[tuple(config.BACKENDS)]] = list(config.BACKENDS)


class SmokeTestSelection(Strict):
    """The cases of a custom Smoke Test; `fetch` always runs, as every other case needs it."""

    finetune: FinetuneCases | None = None
    # Runs snippets in the Sandbox and checks its limits, from a Pipeline step.
    sandbox: bool = False
    # Uploads a tiny model and trains an sft/lora/hf Phase from it.
    uploaded_model: bool = False
    # Serves the Base Model, the tiny model and, with a finetune case, its Adapter on Endpoints;
    # with a finetune case, also runs the `serve` Stage.
    serving: bool = False
    # Evaluates the Base Model on a benchmark and on a coding benchmark of each harness, and, with
    # a finetune case, its Adapter.
    evaluate: bool = False
    # Distills prompts with the Base Model as Teacher and a tool, then trains on its replies.
    distill: bool = False
    # Trains an sft Phase, then a dpo Phase that continues its Adapter.
    chain: bool = False
    # rsLoRA, merged QLoRA and DoRA, and `full` after an Adapter; `serving` serves the merged one.
    weights: bool = False
    # Stops an sft Phase after its first Checkpoint, then continues it from there.
    resume: bool = False


COMPLETE_SMOKE_TEST = SmokeTestSelection(
    finetune=FinetuneCases(),
    sandbox=True,
    uploaded_model=True,
    serving=True,
    evaluate=True,
    distill=True,
    chain=True,
    weights=True,
    resume=True,
)


def finetune_cases(selection: SmokeTestSelection) -> dict[str, tuple[tuple[dict, ...], str]]:
    """Case name -> its Phases, each its algorithm and method, and its backend."""
    chosen = selection.finetune or FinetuneCases(phases=[], methods=[], backends=[])
    cases = {
        f"{phase}-{method}-{backend}": (({"algorithm": phase, "method": method},), backend)
        for phase in chosen.phases
        for method in chosen.methods
        for backend in chosen.backends
        if method in config.BACKENDS[backend]
    }
    adapter_case = first_adapter_case(cases)
    if selection.uploaded_model:
        cases[config.SMOKE_TEST_UPLOADED_MODEL_CASE] = config.SMOKE_TEST_TRAINING
    if selection.distill:
        cases[config.SMOKE_TEST_DISTILL_CASE] = config.SMOKE_TEST_TRAINING
    if selection.chain:
        cases[config.SMOKE_TEST_CHAIN_CASE] = config.SMOKE_TEST_CHAIN
    if selection.weights:
        cases.update(config.SMOKE_TEST_WEIGHT_CASES)
    if selection.resume:
        cases[config.SMOKE_TEST_RESUME_CASE] = config.SMOKE_TEST_TRAINING
    if selection.serving and adapter_case:
        cases[config.SMOKE_TEST_SERVE_STAGE_CASE] = cases[adapter_case]
    return cases


def first_adapter_case(trainings: dict) -> str | None:
    """The first case training one Phase on the Base Model that registers an Adapter."""
    return next(
        (
            case
            for case, (phases, _) in trainings.items()
            if case not in config.SMOKE_TEST_SPECIAL_FINETUNE_CASES and keeps_adapter(phases[-1])
        ),
        None,
    )


def keeps_adapter(phase: dict | None) -> bool:
    return phase is not None and phase["method"] != "full" and phase.get("output") != "merged"


def finetune_case_request(
    case: str, smoke_test: str, starting_model: dict, phases: tuple, backend: str
) -> dict:
    """The Pipeline Request of one case, each Phase on its bundled Dataset or `@distill`."""
    stages = {}
    datasets = [f"dataset:{smoke_test}-{phase['algorithm']}" for phase in phases]
    if case == config.SMOKE_TEST_SERVE_STAGE_CASE:
        stages["serve"] = {}
    if case == config.SMOKE_TEST_DISTILL_CASE:
        stages["distill"] = {
            "dataset": f"dataset:{smoke_test}-distill",
            "teacher": config.SMOKE_TEST_BASE_MODEL,
            "tools": config.SMOKE_TEST_DISTILL_TOOLS,
            "max_tokens": config.SMOKE_TEST_DISTILL_MAX_TOKENS,
            "temperature": 0,
        }
        datasets = [config.DISTILL_OUTPUT]
    return {
        **stages,
        "name": f"{smoke_test}-{case}",
        "finetune": {
            **starting_model,
            "backend": backend,
            "phases": [
                phase_request(earlier, phase, dataset)
                for earlier, phase, dataset in zip((None, *phases), phases, datasets, strict=False)
            ],
        },
    }


def phase_request(earlier: dict | None, phase: dict, dataset: str) -> dict:
    """A case's Phase, after the `earlier` one, with the Smoke Test's settings for its algorithm."""
    algorithm = config.ALGORITHMS[phase["algorithm"]]
    length = {algorithm["length_setting"]: config.SMOKE_TEST_PHASE["length"]}
    return {
        "algorithm": phase["algorithm"],
        "dataset": dataset,
        "method": phase["method"],
        "output": phase.get("output", "adapter"),
        "settings": {**config.SMOKE_TEST_PHASE["settings"], **length},
        # After a kept Adapter, a Phase continues it, so only the others set `lora`.
        **(
            {"lora": {**config.SMOKE_TEST_PHASE["lora"], **phase.get("lora", {})}}
            if phase["method"] != "full" and not keeps_adapter(earlier)
            else {}
        ),
        **({"teacher": config.SMOKE_TEST_BASE_MODEL} if algorithm["learns_from_teacher"] else {}),
    }


def evaluate_cases(selection: SmokeTestSelection, trainings: dict) -> dict[str, str | None]:
    """Evaluate case -> the finetune case whose Adapter it evaluates; None for the Base Model."""
    if not selection.evaluate:
        return {}
    adapter, made_by = config.SMOKE_TEST_ADAPTER_EVALUATE_CASE, first_adapter_case(trainings)
    return {
        case: made_by if case == adapter else None
        for case in config.SMOKE_TEST_EVALUATE_CASES
        if case != adapter or made_by
    }


def evaluate_case_request(case: str, smoke_test: str, model: str) -> dict:
    """The Pipeline Request of one evaluate case: its benchmark or performance run on the model."""
    evaluate = {"model": model, **config.SMOKE_TEST_EVALUATE_CASES[case]}
    return {"name": f"{smoke_test}-{case}", "evaluate": evaluate}


def run_order(sandbox: bool, trainings: dict, evaluations: dict) -> list[str]:
    """The cases run as Kubeflow nodes, in the order they run."""
    # The `serve` Stage case comes last, as its serve step runs only once its training passed.
    serve_stage = [case for case in trainings if case == config.SMOKE_TEST_SERVE_STAGE_CASE]
    finetunes = [case for case in trainings if case not in serve_stage]
    sandbox_case = [config.SMOKE_TEST_SANDBOX_CASE] if sandbox else []
    return ["fetch", *sandbox_case, *finetunes, *evaluations, *serve_stage]


def serving_cases(
    selection: SmokeTestSelection,
    trainings: dict,
    smoke_test: str,
    base_model: str,
    uploaded_model: str | None,
) -> dict[str, dict]:
    """Serving case -> the model its Endpoint serves, and the finetune case making it, if any."""
    if not selection.serving:
        return {}
    base, full_weights, adapter, merged = config.SMOKE_TEST_SERVING_CASES
    cases = {base: {"model": base_model}, full_weights: {"model": uploaded_model}}
    made_by = {adapter: first_adapter_case(trainings)}
    if config.SMOKE_TEST_MERGED_CASE in trainings:
        made_by[merged] = config.SMOKE_TEST_MERGED_CASE
    for case, finetune_case in made_by.items():
        if finetune_case:
            model = model_reference(f"{smoke_test}-{finetune_case}", 1)
            cases[case] = {"model": model, "made_by": finetune_case}
    return cases


def has_pending_serving_cases(cases: dict[str, str]) -> bool:
    return any(cases.get(case) == "pending" for case in config.SMOKE_TEST_SERVING_CASES)


def case_results(cases: dict[str, str], run: KubeflowRun | None, finished: bool) -> dict:
    """Each case's result from its step, else its last one; once finished, pending ones failed."""
    states = run.step_states if run else {}
    results = {
        case: config.SMOKE_TEST_CASE_RESULTS.get(states.get(case), last)
        for case, last in cases.items()
    }
    if finished:
        return {case: "failed" if r == "pending" else r for case, r in results.items()}
    return results
