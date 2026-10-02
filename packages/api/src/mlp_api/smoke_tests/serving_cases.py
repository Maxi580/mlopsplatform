import logging

from sqlalchemy import select

from mlp_api.endpoints.lifecycle import list_endpoints, start_endpoint, stop_endpoint
from mlp_api.pipelines.lifecycle import pipeline, set_pipeline, unfinished
from mlp_api.smoke_tests.lifecycle import is_smoke_test
from mlp_core import config
from mlp_core.endpoint_spec import EndpointSpec

logger = logging.getLogger(__name__)


def run_serving_cases(state) -> None:
    """Advances every running Smoke Test's pending serving cases by one step."""
    # 1. The running Smoke Tests, and each name's newest Endpoint, as stopped ones keep their row.
    with state.engine.connect() as connection:
        rows = connection.execute(select(pipeline).where(is_smoke_test, unfinished)).all()
    endpoints = {}
    for found in list_endpoints(state.engine):
        endpoints.setdefault(found["name"], found)

    # 2. Each pending serving case's next result.
    for row in rows:
        cases = dict(row.cases)
        for case in config.SMOKE_TEST_SERVING_CASES:
            if cases.get(case) == "pending":
                name = f"{row.name}-{case}"
                cases[case] = serving_case_result(state, name, row.request[case], cases, endpoints)
        set_pipeline(state.engine, row.id, cases=cases)

        # 3. The `serve` Stage case's Endpoint only had to start; it would hold a GPU from the rest.
        serve_stage = config.SMOKE_TEST_SERVE_STAGE_CASE
        endpoint = endpoints.get(f"{row.name}-{serve_stage}")
        has_result = cases.get(serve_stage, "pending") != "pending"
        if has_result and endpoint and endpoint["status"] != "stopped":
            stop_endpoint(state.engine, state.cluster, endpoint["name"])


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
            start_endpoint(state, name, EndpointSpec(model=serving["model"]))
        except Exception:
            logger.exception("Smoke Test Endpoint %s did not start", name)
            return "failed"
        return "pending"

    # 3. Passed once vLLM is ready; either way, its Endpoint then stops.
    if endpoint["status"] == "pending":
        return "pending"
    if endpoint["status"] != "stopped":
        stop_endpoint(state.engine, state.cluster, name)
    return "passed" if endpoint["status"] == "running" else "failed"
