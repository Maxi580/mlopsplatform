import logging
import time

from mlp_api.endpoints.request_traces import request_content, span_attributes
from mlp_core import config

logger = logging.getLogger(__name__)


def is_traced_in_full(model_registry, trace_id: str, request: dict) -> bool:
    """Whether the trace holds the prompt, an answer and vLLM's span within the timeout."""
    # The Endpoint Proxy and vLLM export their spans in the background.
    deadline = time.monotonic() + config.SMOKE_TEST_TRACE_TIMEOUT.total_seconds()
    while True:
        for trace in model_registry.batch_get_traces([trace_id]):
            content = request_content(trace)
            if (
                content["messages"] == request["messages"]
                and content["answer"]
                and span_attributes(trace, config.REQUEST_TRACE_SPAN) is not None
            ):
                return True
        if time.monotonic() > deadline:
            logger.error("Smoke Test trace %s did not arrive in full", trace_id)
            return False
        time.sleep(config.SMOKE_TEST_TRACE_POLL_INTERVAL.total_seconds())
