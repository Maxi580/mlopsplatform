import logging
from datetime import UTC, datetime

from sqlalchemy import select

from mlp_api.endpoints.feedback import record_feedback
from mlp_api.endpoints.lifecycle import (
    list_endpoints,
    start_endpoint,
    stop_endpoint,
    traces_experiment_id,
)
from mlp_api.endpoints.stats import curated_values, fetch_endpoint_stats
from mlp_api.pipelines.lifecycle import pipeline, set_pipeline, unfinished
from mlp_api.smoke_tests.judge_case import judge_case_result
from mlp_api.smoke_tests.lifecycle import is_smoke_test
from mlp_api.smoke_tests.traces import is_traced_in_full
from mlp_core import config
from mlp_core.endpoint_spec import EndpointSpec

logger = logging.getLogger(__name__)


def run_serving_cases(state) -> None:
    """Advances every running Smoke Test's pending serving cases by one step."""
    # 1. The running Smoke Tests, and each name's newest Endpoint; older databases have several.
    with state.engine.connect() as connection:
        rows = connection.execute(select(pipeline).where(is_smoke_test, unfinished)).all()
    endpoints = {}
    for found in list_endpoints(state.engine, state.cluster.endpoint_environment.domain):
        endpoints.setdefault(found["name"], found)

    # 2. Each pending serving case's next result.
    for row in rows:
        cases = dict(row.cases)
        for case in config.SMOKE_TEST_SERVING_CASES:
            if cases.get(case) != "pending":
                continue
            if case == config.SMOKE_TEST_JUDGE_CASE:
                cases[case] = judge_case_result(state, row.name, row.request[case])
            else:
                name = f"{row.name}-{case}"
                cases[case] = serving_case_result(state, name, row.request[case], cases, endpoints)
        set_pipeline(state.engine, row.id, cases=cases)


def serving_case_result(state, name: str, serving: dict, cases: dict, endpoints: dict) -> str:
    """Starts, waits for or stops the case's Endpoint; returns the case's result so far."""
    # 1. Its model exists once the finetune case making it passed.
    made_by = cases.get(serving.get("made_by"))
    if made_by in ("pending", "failed"):
        return made_by

    # 2. Its Endpoint, started once.
    endpoint = endpoints.get(name)
    if endpoint is None:
        try:
            spec = EndpointSpec(model=serving["model"], speculative=serving.get("speculative"))
            start_endpoint(state, name, spec)
        except Exception:
            logger.exception("Smoke Test Endpoint %s did not start", name)
            return "failed"
        return "pending"

    # 3. Passed once vLLM is ready, answers a real chat request and its stats and trace show it;
    # then it stops.
    if endpoint["status"] == "pending":
        return "pending"
    passed = endpoint["status"] == "running" and counts_a_chat_request(state, name)
    if endpoint["status"] != "stopped":
        stop_endpoint(state.engine, state.cluster, name)
    return "passed" if passed else "failed"


def counts_a_chat_request(state, name: str) -> bool:
    """Whether the Endpoint answers a chat request through the Endpoint Proxy, after which its
    stats show it finished and its Request Trace holds it in full."""
    # 1. The request, through the proxy.
    try:
        experiment_id = traces_experiment_id(state, name)
        request = config.SMOKE_TEST_CHAT_REQUEST
        trace_id = state.cluster.send_chat_request(name, request, experiment_id)
    except Exception:
        logger.exception("Smoke Test Endpoint %s did not answer a chat request", name)
        return False

    # 2. Its stats.
    stats = fetch_endpoint_stats(state.cluster, [name])[name]
    if stats is None:
        return False
    values = curated_values(stats)
    finished = values.get("finished", {}).get("value") or 0
    if finished < 1 or not values.get("generation_tokens", {}).get("value"):
        return False

    # 3. Its trace, once both spans arrived, which then takes a 👍.
    if not is_traced_in_full(state.model_registry, trace_id, request):
        return False
    try:
        record_feedback(state, name, trace_id, "up", datetime.now(UTC))
    except Exception:
        logger.exception("Smoke Test Endpoint %s's trace took no Feedback", name)
        return False
    info = state.model_registry.trace_info(trace_id)
    rated = [a for a in info.get("assessments", []) if a["assessment_name"] == "rating"]
    return info["tags"].get("rating") == "up" and bool(rated)
