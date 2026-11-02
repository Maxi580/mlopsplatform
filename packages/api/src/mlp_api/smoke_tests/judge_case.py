import logging

from mlp_api.endpoints.lifecycle import find_endpoint
from mlp_api.judge.lifecycle import save_judge_settings
from mlp_api.judge.settings import JudgeSettings, judge_settings
from mlp_core import config
from mlp_core.endpoint_spec import EndpointSpec

logger = logging.getLogger(__name__)


def judge_case_result(state, case: dict) -> str:
    """Runs the Judge on the case's model until it answers on its Service, then switches judging
    back as it was; returns the case's result so far."""
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
    if found is None or found.status == "pending":
        return "pending"
    passed = found.status == "running" and judge_answers(state)

    # 3. Judging as it was; switched off, the Judge is gone.
    restore_judging(state, case)
    restored = JudgeSettings.model_validate(case["restore"])
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


def restore_judging(state, case: dict) -> None:
    """Judging as it was before the case, unless it was changed since the case switched it on."""
    settings = judge_settings(state.engine)
    if settings.enabled and settings.spec == EndpointSpec(model=case["model"]):
        save_judge_settings(state, JudgeSettings.model_validate(case["restore"]))
