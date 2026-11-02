import threading

from mlp_api.endpoints.lifecycle import (
    delete_endpoint,
    find_endpoint,
    start_endpoint,
    stop_endpoint,
)
from mlp_api.judge.settings import JudgeSettings, judge_settings, store_judge_settings
from mlp_core import config

# Saving the settings and the reconciler never start the Judge twice.
judge_lock = threading.Lock()


def save_judge_settings(state, settings: JudgeSettings) -> None:
    """Stores the settings and starts, replaces or removes the Judge to match."""
    # A model that can't be served raises ValueError, after which judging stays off.
    with judge_lock:
        # 1. A new spec, or judging off, ends the Judge that runs; a new rubric alone doesn't.
        if not settings.enabled or settings.spec != judge_settings(state.engine).spec:
            remove_judge(state)
        store_judge_settings(state.engine, settings)

        # 2. Started at once, so a model that can't be served is reported now.
        try:
            keep_judge_running(state)
        except ValueError:
            store_judge_settings(state.engine, settings.model_copy(update={"enabled": False}))
            raise


def reconcile_judge(state) -> None:
    """Keeps the Judge running with its settings' spec while judging is enabled, else none."""
    with judge_lock:
        keep_judge_running(state)


def keep_judge_running(state) -> None:
    # A Judge gone after a crash or an upgrade is started again; one no longer wanted is removed.
    settings = judge_settings(state.engine)
    found = find_endpoint(state.engine, config.JUDGE_ENDPOINT_NAME)
    if found is not None and (found.status == "stopped" or not settings.enabled):
        remove_judge(state)
        found = None
    if settings.enabled and found is None:
        start_endpoint(state, config.JUDGE_ENDPOINT_NAME, settings.spec)


def remove_judge(state) -> None:
    """Stops the Judge, if one runs, and deletes it."""
    found = find_endpoint(state.engine, config.JUDGE_ENDPOINT_NAME)
    if found is None:
        return
    if found.status != "stopped":
        stop_endpoint(state.engine, state.cluster, config.JUDGE_ENDPOINT_NAME)
    delete_endpoint(state, config.JUDGE_ENDPOINT_NAME)


def judge_summary(state) -> dict:
    """The settings, and the Judge's status and pinned model; both None while there is none."""
    found = find_endpoint(state.engine, config.JUDGE_ENDPOINT_NAME)
    settings = judge_settings(state.engine)
    return {
        **settings.model_dump(mode="json"),
        # As saved, without the options left at their defaults.
        "spec": settings.spec and settings.spec.model_dump(mode="json", exclude_defaults=True),
        "status": found and found.status,
        "model": found and found.spec["model"],
    }
