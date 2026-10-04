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


COMPLETE_SMOKE_TEST = SmokeTestSelection(
    finetune=FinetuneCases(),
    sandbox=True,
    uploaded_model=True,
    serving=True,
    evaluate=True,
    distill=True,
)


def finetune_cases(selection: SmokeTestSelection) -> dict[str, tuple[str, str, str]]:
    """Case name -> its Phase algorithm, method and backend."""
    chosen = selection.finetune or FinetuneCases(phases=[], methods=[], backends=[])
    cases = {
        f"{phase}-{method}-{backend}": (phase, method, backend)
        for phase in chosen.phases
        for method in chosen.methods
        for backend in chosen.backends
        if method in config.BACKENDS[backend]
    }
    first_training = next(iter(cases.values()), None)
    if selection.uploaded_model:
        cases[config.SMOKE_TEST_UPLOADED_MODEL_CASE] = config.SMOKE_TEST_TRAINING
    if selection.distill:
        cases[config.SMOKE_TEST_DISTILL_CASE] = config.SMOKE_TEST_TRAINING
    if selection.serving and first_training:
        cases[config.SMOKE_TEST_SERVE_STAGE_CASE] = first_training
    return cases


def finetune_case_request(
    case: str, smoke_test: str, starting_model: dict, phase: str, method: str, backend: str
) -> dict:
    """The Pipeline Request of one case, training on its Phase's bundled Dataset or `@distill`."""
    stages, dataset = {}, f"dataset:{smoke_test}-{phase}"
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
        dataset = config.DISTILL_OUTPUT
    return {
        **stages,
        "name": f"{smoke_test}-{case}",
        "finetune": {
            **starting_model,
            "backend": backend,
            "phases": [
                {
                    "algorithm": phase,
                    "dataset": dataset,
                    "method": method,
                    **config.SMOKE_TEST_PHASE,
                }
            ],
        },
    }


def evaluate_cases(selection: SmokeTestSelection, trainings: dict) -> dict[str, str | None]:
    """Evaluate case -> the finetune case whose Adapter it evaluates; None for the Base Model."""
    if not selection.evaluate:
        return {}
    adapter = config.SMOKE_TEST_ADAPTER_EVALUATE_CASE
    finetunes = [c for c in trainings if c not in config.SMOKE_TEST_SPECIAL_FINETUNE_CASES]
    return {
        case: finetunes[0] if case == adapter else None
        for case in config.SMOKE_TEST_EVALUATE_CASES
        if case != adapter or finetunes
    }


def evaluate_case_request(case: str, smoke_test: str, model: str) -> dict:
    """The Pipeline Request of one evaluate case: a few samples of its benchmark."""
    return {
        "name": f"{smoke_test}-{case}",
        "evaluate": {
            "model": model,
            "benchmarks": [config.SMOKE_TEST_EVALUATE_CASES[case]],
            "limit": config.SMOKE_TEST_EVALUATE_LIMIT,
        },
    }


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
    base, full_weights, adapter = config.SMOKE_TEST_SERVING_CASES
    cases = {base: {"model": base_model}, full_weights: {"model": uploaded_model}}
    finetunes = [case for case in trainings if case != config.SMOKE_TEST_UPLOADED_MODEL_CASE]
    if finetunes:
        made_by = finetunes[0]
        model = model_reference(f"{smoke_test}-{made_by}", 1)
        cases[adapter] = {"model": model, "made_by": made_by}
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
