import logging
from datetime import UTC, datetime

from mlp_api.endpoints.lifecycle import (
    find_endpoint,
    start_endpoint,
    stop_endpoint,
    traces_experiment_id,
)
from mlp_api.judge.judging import Judging, judge_once, set_endpoint_judging
from mlp_api.judge.lifecycle import save_judge_settings
from mlp_api.judge.settings import JudgeSettings, judge_settings
from mlp_api.smoke_tests.traces import is_traced_in_full
from mlp_core import config
from mlp_core.endpoint_spec import EndpointSpec, without_stored_quantization

logger = logging.getLogger(__name__)


def judge_case_result(state, smoke_test: str, case: dict) -> str:
    """The judge case's result so far, as it runs the Judge, then switches judging back."""
    # 1. Judging on, with the case's model.
    settings = judge_settings(state.engine)
    spec = EndpointSpec(model=case["model"])
    if not (settings.enabled and settings.spec == spec):
        try:
            save_judge_settings(state, settings.model_copy(update={"enabled": True, "spec": spec}))
        except Exception:
            logger.exception("The Smoke Test's Judge did not start")
            return "failed"
        return "pending"

    # 2. Once vLLM is ready, a chat request on its Service.
    found = find_endpoint(state.engine, config.JUDGE_ENDPOINT_NAME)
    if found is None or found.status in config.ENDPOINT_STARTING_STATUSES:
        return "pending"
    passed = found.status == "running" and judge_answers(state)

    # 3. An Endpoint judged every answer, once one runs beside the Judge.
    if passed and state.settings.gpu_count >= 2:
        judged = judged_answer_result(state, f"{smoke_test}-{config.SMOKE_TEST_JUDGE_CASE}", case)
        if judged == "pending":
            return "pending"
        passed = judged == "passed"

    # 4. Judging as it was; switched off, the Judge is gone.
    restore_judging(state, case)
    restored = settings_to_restore(case)
    gone = find_endpoint(state.engine, config.JUDGE_ENDPOINT_NAME) is None
    return "passed" if passed and (restored.enabled or gone) else "failed"


def judge_answers(state) -> bool:
    try:
        answer = state.cluster.ask_endpoint(
            config.JUDGE_ENDPOINT_NAME,
            config.SMOKE_TEST_CHAT_REQUEST,
            config.SMOKE_TEST_CHAT_TIMEOUT.total_seconds(),
        )
        return bool(answer["choices"][0]["message"]["content"])
    except Exception:
        logger.exception("The Smoke Test's Judge did not answer a chat request")
        return False


def judged_answer_result(state, name: str, case: dict) -> str:
    """The result so far of an Endpoint, judged on every answer, getting its answer scored."""
    # 1. The Endpoint, started once.
    found = find_endpoint(state.engine, name)
    if found is None:
        try:
            start_endpoint(state, name, EndpointSpec(model=case["model"]), smoke_test=True)
            judging = Judging(judged=True, every=1, max_per_hour=1)
            set_endpoint_judging(state.engine, name, judging, datetime.now(UTC))
        except Exception:
            logger.exception("The Smoke Test's judged Endpoint %s did not start", name)
            return "failed"
        return "pending"
    if found.status in config.ENDPOINT_STARTING_STATUSES:
        return "pending"

    # 2. Its answer, traced in full, then scored by a judging round.
    passed = found.status == "running" and gets_a_judge_score(state, name)
    if found.status != "stopped":
        stop_endpoint(state.engine, state.cluster, name)
    return "passed" if passed else "failed"


def gets_a_judge_score(state, name: str) -> bool:
    try:
        request = config.SMOKE_TEST_CHAT_REQUEST
        trace_id = state.cluster.send_chat_request(name, request, traces_experiment_id(state, name))
        if not is_traced_in_full(state.model_registry, trace_id, request):
            return False
        judge_once(state, datetime.now(UTC))
    except Exception:
        logger.exception("The Smoke Test's judged Endpoint %s got no Judge Score", name)
        return False
    return config.JUDGE_SCORE_TAG in state.model_registry.trace_info(trace_id).get("tags", {})


def restore_judging(state, case: dict) -> None:
    """Judging as it was before the case, unless it was changed since the case switched it on."""
    settings = judge_settings(state.engine)
    if settings.enabled and settings.spec == EndpointSpec(model=case["model"]):
        save_judge_settings(state, settings_to_restore(case))


def settings_to_restore(case: dict) -> JudgeSettings:
    stored = without_stored_quantization(case["restore"])
    # A Smoke Test running across the upgrade stored the system prompt under its old name.
    if "rubric" in stored:
        stored["system_prompt"] = stored.pop("rubric")
    return JudgeSettings.model_validate(stored)
