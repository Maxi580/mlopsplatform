import logging
import time

from mlp_api.endpoints.request_traces import span_attributes
from mlp_core import config

logger = logging.getLogger(__name__)


def is_traced_in_full(model_registry, trace_id: str, request: dict) -> bool:
    """Whether the trace holds the request's prompt, an answer, and vLLM's span inside it, within
    SMOKE_TEST_TRACE_TIMEOUT, as the proxy and vLLM export in the background."""
    deadline = time.monotonic() + config.SMOKE_TEST_TRACE_TIMEOUT.total_seconds()
    while True:
        for trace in model_registry.batch_get_traces([trace_id]):
            root = span_attributes(trace, config.REQUEST_TRACE_ROOT_SPAN) or {}
            answer = (root.get(config.MLFLOW_SPAN_OUTPUTS) or {}).get("choices") or [{}]
            if (
                (root.get(config.MLFLOW_SPAN_INPUTS) or {}).get("messages") == request["messages"]
                and answer[0].get("message", {}).get("content")
                and span_attributes(trace, config.REQUEST_TRACE_SPAN) is not None
            ):
                return True
        if time.monotonic() > deadline:
            logger.error("Smoke Test trace %s did not arrive in full", trace_id)
            return False
        time.sleep(config.SMOKE_TEST_TRACE_POLL_INTERVAL.total_seconds())
