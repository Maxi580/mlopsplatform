from datetime import UTC, datetime
from urllib.parse import urlsplit

import pytest

from mlp_core import api_paths

from .conftest import CHAT_ANSWER, record_request_trace
from .test_endpoints import (
    qwen_on_the_hub,  # noqa: F401
    start,
)

REQUEST = {"messages": [{"role": "user", "content": "Say hi."}]}


@pytest.fixture
def chat(logged_in_api, qwen_on_the_hub, model_registry):  # noqa: F811
    """The started Endpoint `chat`, with one Request Trace `tr-1`."""
    started = start(logged_in_api).json()
    now = datetime.now(UTC)
    record_request_trace(model_registry, "endpoint/chat", "tr-1", now, 0.5, REQUEST, CHAT_ANSWER)
    return started


def vote(api, endpoint, rating, trace_id="tr-1", key=None):
    uuid = urlsplit(endpoint["url"]).path.split("/")[2]
    headers = {"Authorization": f"Bearer {key or endpoint['key']}"}
    body = {"trace_id": trace_id, "rating": rating}
    return api.post(api_paths.ENDPOINT_FEEDBACK.format(uuid=uuid), json=body, headers=headers)


def counted(model_registry) -> list[tuple[str, float]]:
    return [(key, value) for run in model_registry.metrics.values() for key, value, _ in run]


def test_a_vote_lands_on_the_trace_as_assessment_and_tag_and_in_the_history(
    api, chat, model_registry
):
    api.cookies.clear()

    response = vote(api, chat, "up")

    assert response.status_code == 200, response.text
    info = model_registry.trace_info("tr-1")
    assert info["tags"]["rating"] == "up"
    [assessment] = info["assessments"]
    assert assessment["assessment_name"] == "rating"
    assert assessment["feedback"] == {"value": "up"}
    assert assessment["source"]["source_type"] == "HUMAN"
    assert counted(model_registry) == [("feedback_up", 1)]


def test_a_changed_vote_replaces_the_old_one_and_takes_back_its_count(
    logged_in_api, chat, model_registry
):
    vote(logged_in_api, chat, "up")
    vote(logged_in_api, chat, "up")

    vote(logged_in_api, chat, "down")

    info = model_registry.trace_info("tr-1")
    assert info["tags"]["rating"] == "down"
    assert [a["feedback"]["value"] for a in info["assessments"]] == ["down"]
    assert counted(model_registry) == [
        ("feedback_up", 1),
        ("feedback_down", 1),
        ("feedback_up", -1),
    ]


def test_a_vote_needs_the_endpoint_key(logged_in_api, chat, model_registry):
    other = start(logged_in_api, name="other").json()

    assert vote(logged_in_api, chat, "up", key="wrong").status_code == 401
    assert vote(logged_in_api, chat, "up", key=other["key"]).status_code == 401
    assert "rating" not in model_registry.trace_info("tr-1")["tags"]


def test_a_vote_on_another_endpoints_trace_is_not_found(logged_in_api, chat, model_registry):
    other = start(logged_in_api, name="other").json()

    assert vote(logged_in_api, other, "up").status_code == 404
    assert vote(logged_in_api, chat, "up", trace_id="tr-unknown").status_code == 404


def test_a_vote_is_up_or_down(logged_in_api, chat):
    assert vote(logged_in_api, chat, "meh").status_code == 422


def test_the_history_sums_the_votes_per_bucket(logged_in_api, chat, model_registry):
    vote(logged_in_api, chat, "up")
    record_request_trace(
        model_registry, "endpoint/chat", "tr-2", datetime.now(UTC), 0.5, REQUEST, CHAT_ANSWER
    )
    vote(logged_in_api, chat, "up", trace_id="tr-2")

    history = logged_in_api.get(api_paths.ENDPOINT_HISTORY.format(name="chat")).json()

    [(_, ups)] = history["feedback"]["up"]
    assert ups == 2
    assert history["feedback"]["down"] == []


def test_the_route_sends_feedback_to_the_api_behind_the_endpoint_key(logged_in_api, chat, cluster):
    uuid = urlsplit(chat["url"]).path.split("/")[2]

    feedback, _ = cluster.endpoints["chat"]["route"]["spec"]["routes"]

    assert feedback["match"] == f"Path(`/serving/{uuid}/v1/feedback`)"
    assert feedback["middlewares"] == [{"name": "endpoint-key"}]
    assert feedback["services"] == [{"name": "api", "port": "http"}]
