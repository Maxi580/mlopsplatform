import json
from datetime import UTC, datetime, timedelta

import pytest

from mlp_api.judge.judging import judge_once
from mlp_core import api_paths, config

from .conftest import CHAT_ANSWER, record_request_trace
from .test_endpoint_stats import SERVED
from .test_endpoints import endpoints, qwen_on_the_hub, start  # noqa: F401
from .test_judge import ON, rounds, save

REQUEST = {
    "messages": [{"role": "user", "content": "Say hi."}],
    "tools": [{"type": "function", "function": {"name": "wave"}}],
}
REASONED = {
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hi!",
                "reasoning_content": "Greet back.",
                "tool_calls": [{"id": "c", "function": {"name": "wave", "arguments": "{}"}}],
            },
        }
    ],
}


def judgment(score, reason="Polite.") -> dict:
    content = json.dumps({"score": score, "reason": reason})
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]}


def opt_in(api, name="chat", **judging):
    body = {"judged": True, "every": 1, "max_per_hour": 60, **judging}
    return api.put(api_paths.ENDPOINT_JUDGING.format(name=name), json=body)


@pytest.fixture
def judge_running(logged_in_api, qwen_on_the_hub, cluster):  # noqa: F811
    """A running Judge, which scores 4, and the Endpoint `chat`."""
    save(logged_in_api)
    cluster.endpoint_states["judge"] = "running"
    cluster.metrics["judge"] = SERVED
    cluster.answers["judge"] = judgment(4)
    rounds(logged_in_api)
    start(logged_in_api)


def arrive(model_registry, trace_id, seconds=0, tags=None, answer=REASONED):
    arrived = datetime.now(UTC) + timedelta(seconds=seconds)
    record_request_trace(
        model_registry, "endpoint/chat", trace_id, arrived, 0.5, REQUEST, answer, tags
    )


def judge(api):
    judge_once(api.app.state, datetime.now(UTC) + timedelta(seconds=30))


def tags(model_registry, trace_id) -> dict:
    return model_registry.trace_info(trace_id)["tags"]


def test_an_endpoint_opts_in_to_judging_with_every_n_and_a_cap(logged_in_api, judge_running):
    response = opt_in(logged_in_api, every=5, max_per_hour=20)

    assert response.status_code == 200, response.text
    assert response.json()["judging"] == {"judged": True, "every": 5, "max_per_hour": 20}
    chat = [e for e in endpoints(logged_in_api) if e["name"] == "chat"][0]
    assert chat["judging"]["judged"] is True


@pytest.mark.parametrize("judging", [{"every": 0}, {"max_per_hour": 0}])
def test_judging_takes_only_positive_counts(logged_in_api, judge_running, judging):
    assert opt_in(logged_in_api, **judging).status_code == 422


def test_the_judge_itself_and_unknown_endpoints_cant_opt_in(logged_in_api, judge_running):
    assert opt_in(logged_in_api, name="judge").status_code == 409
    assert opt_in(logged_in_api, name="nope").status_code == 404


def test_the_judge_scores_an_answer_with_the_rubric_conversation_reasoning_and_tool_calls(
    logged_in_api, judge_running, cluster, model_registry
):
    opt_in(logged_in_api)
    # Not after now, so the history shows its score.
    arrive(model_registry, "tr-1")

    judge(logged_in_api)

    [(name, request)] = cluster.asked
    assert name == "judge"
    assert request["response_format"] == config.JUDGE_RESPONSE_FORMAT
    [prompt] = request["messages"]
    text = prompt["content"]
    assert text.startswith("Be strict.")
    assert "user: Say hi." in text and '"wave"' in text
    assert "Greet back." in text.split("reasoning")[1]
    assert "Hi!" in text and "Tool calls" in text
    assert tags(model_registry, "tr-1")["judge_score"] == "4"
    [assessment] = model_registry.trace_info("tr-1")["assessments"]
    assert assessment["assessment_name"] == "judge"
    assert assessment["feedback"] == {"value": 4}
    assert assessment["rationale"] == "Polite."
    assert assessment["source"]["source_type"] == "LLM_JUDGE"
    history = logged_in_api.get(api_paths.ENDPOINT_HISTORY.format(name="chat")).json()
    assert [value for _, value in history["judge_score"]] == [4]


def test_every_nth_trace_is_scored_and_the_rest_are_passed_over_once(
    logged_in_api, judge_running, model_registry
):
    opt_in(logged_in_api, every=2)
    for index in range(1, 5):
        arrive(model_registry, f"tr-{index}", seconds=index)

    judge(logged_in_api)
    judge(logged_in_api)

    found = [tags(model_registry, f"tr-{index}") for index in range(1, 5)]
    assert [t.get("judge_score") for t in found] == [None, "4", None, "4"]
    assert [t.get("judge_status") for t in found] == ["unselected", None, "unselected", None]


def test_the_count_survives_a_round_so_every_nth_spans_rounds(
    logged_in_api, judge_running, model_registry
):
    opt_in(logged_in_api, every=2)
    arrive(model_registry, "tr-1", seconds=1)
    judge(logged_in_api)
    arrive(model_registry, "tr-2", seconds=2)

    judge(logged_in_api)

    assert tags(model_registry, "tr-2")["judge_score"] == "4"


def test_the_hourly_cap_stops_judging_until_the_next_hour(
    logged_in_api, judge_running, model_registry
):
    opt_in(logged_in_api, max_per_hour=1)
    arrive(model_registry, "tr-1", seconds=1)
    arrive(model_registry, "tr-2", seconds=2)

    judge(logged_in_api)

    assert "judge_score" in tags(model_registry, "tr-1")
    assert not {"judge_score", "judge_status"} & set(tags(model_registry, "tr-2"))
    judge_once(logged_in_api.app.state, datetime.now(UTC) + timedelta(hours=1))
    assert tags(model_registry, "tr-2")["judge_score"] == "4"


@pytest.mark.parametrize(
    "answer",
    [judgment(9), {"choices": [{"message": {"content": "Great answer!"}}]}],
    ids=["out-of-range", "unparseable"],
)
def test_a_judgment_that_cant_be_used_marks_the_trace_failed_for_good(
    logged_in_api, judge_running, cluster, model_registry, answer
):
    cluster.answers["judge"] = answer
    opt_in(logged_in_api)
    arrive(model_registry, "tr-1", seconds=1)

    judge(logged_in_api)
    asked = len(cluster.asked)
    judge(logged_in_api)

    assert tags(model_registry, "tr-1")["judge_status"] == "failed"
    assert "judge_score" not in tags(model_registry, "tr-1")
    assert len(cluster.asked) == asked


def test_a_failed_request_is_skipped(logged_in_api, judge_running, cluster, model_registry):
    opt_in(logged_in_api)
    arrive(model_registry, "tr-1", seconds=1, tags={"status": "error"}, answer=CHAT_ANSWER)

    judge(logged_in_api)

    assert tags(model_registry, "tr-1")["judge_status"] == "skipped"
    assert cluster.asked == []


def test_traces_before_opting_in_and_endpoints_not_opted_in_are_not_judged(
    logged_in_api, judge_running, cluster, model_registry
):
    arrive(model_registry, "tr-old", seconds=-60)
    judge(logged_in_api)
    opt_in(logged_in_api)

    judge(logged_in_api)

    assert "judge_status" not in tags(model_registry, "tr-old")
    assert cluster.asked == []


def test_nothing_is_judged_while_the_judge_isnt_running(
    logged_in_api,
    qwen_on_the_hub,  # noqa: F811
    cluster,
    model_registry,
):
    save(logged_in_api, **{**ON, "enabled": False})
    start(logged_in_api)
    opt_in(logged_in_api)
    arrive(model_registry, "tr-1", seconds=1)

    judge(logged_in_api)

    assert "judge_status" not in tags(model_registry, "tr-1")
