from typing import Literal

from mlp_api.pipelines.cluster import KubeflowRun
from mlp_core import config
from mlp_core.pipeline_request.schema import METHODS, Strict


class FinetuneCases(Strict):
    """Every combination of these that a backend supports is one case; unnamed means all."""

    phases: list[Literal[tuple(config.ALGORITHMS)]] = list(config.ALGORITHMS)
    methods: list[Literal[METHODS]] = list(METHODS)
    backends: list[Literal[tuple(config.BACKENDS)]] = list(config.BACKENDS)


class SmokeTestSelection(Strict):
    """The cases of a custom Smoke Test; `fetch` always runs, as every other case needs it."""

    finetune: FinetuneCases | None = None


COMPLETE_SMOKE_TEST = SmokeTestSelection(finetune=FinetuneCases())


def finetune_cases(selection: SmokeTestSelection) -> dict[str, tuple[str, str, str]]:
    """Case name -> its Phase algorithm, method and backend."""
    if selection.finetune is None:
        return {}
    chosen = selection.finetune
    return {
        f"{phase}-{method}-{backend}": (phase, method, backend)
        for phase in chosen.phases
        for method in chosen.methods
        for backend in chosen.backends
        if method in config.BACKENDS[backend]
    }


def finetune_case_request(
    case: str, smoke_test: str, base_model: str, phase: str, method: str, backend: str
) -> dict:
    """The Pipeline Request of one case, which trains on the Smoke Test's Dataset for the Phase."""
    return {
        "name": f"{smoke_test}-{case}",
        "finetune": {
            "base_model": base_model,
            "backend": backend,
            "phases": [
                {
                    "algorithm": phase,
                    "dataset": f"dataset:{smoke_test}-{phase}",
                    "method": method,
                    **config.SMOKE_TEST_PHASE,
                }
            ],
        },
    }


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
