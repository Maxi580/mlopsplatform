import json
import logging
import threading
from datetime import UTC, datetime

from sqlalchemy import Engine, select, update

from mlp_api.endpoints.lifecycle import find_endpoint, traces_experiment_id
from mlp_api.endpoints.request_traces import request_content
from mlp_api.endpoints.stats_history import day_run_id, round_ms
from mlp_api.endpoints.table import endpoint
from mlp_api.judge.settings import judge_settings
from mlp_core import config

logger = logging.getLogger(__name__)


def set_endpoint_judging(
    engine: Engine, name: str, judged: bool, every: int, max_per_hour: int, now: datetime
) -> None:
    """Opts the Endpoint in or out of judging; switched on, it counts traces from now on.
    LookupError if there is no Endpoint with the name."""
    found = find_endpoint(engine, name)
    if found is None:
        raise LookupError(f"No Endpoint {name}")
    values = {"judged": judged, "judge_every": every, "judge_max_per_hour": max_per_hour}
    if judged and not found.judged:
        values |= {"judging_since": now, "judge_count": 0}
    with engine.begin() as connection:
        connection.execute(update(endpoint).where(endpoint.c.id == found.id).values(**values))


def judge_forever(state, stop: threading.Event) -> None:
    """Judges every JUDGING_INTERVAL until `stop` is set; a failed round is retried."""
    while not stop.wait(config.JUDGING_INTERVAL.total_seconds()):
        try:
            judge_once(state, datetime.now(UTC))
        except Exception:
            logger.exception("Judging failed")


def judge_once(state, now: datetime) -> None:
    """Scores each judged Endpoint's next selected Request Traces, while the Judge runs."""
    judge = find_endpoint(state.engine, config.JUDGE_ENDPOINT_NAME)
    if judge is None or judge.status != "running":
        return
    rubric = judge_settings(state.engine).rubric
    with state.engine.connect() as connection:
        rows = connection.execute(select(endpoint).where(endpoint.c.judged)).all()
    for row in rows:
        try:
            judge_endpoint(state, row, rubric, now)
        except Exception:
            logger.exception("Judging Endpoint %s's answers failed", row.name)


def judge_endpoint(state, row, rubric: str, now: datetime) -> None:
    """Goes through the Endpoint's traces since it opted in, oldest first: scores every Nth, and
    tags the rest, so none is looked at twice, until the hour's cap."""
    # 1. The traces not looked at yet, whose answer the Endpoint Proxy recorded.
    registry = state.model_registry
    # SQLite, which tests use, keeps no time zone.
    since = row.judging_since.replace(tzinfo=row.judging_since.tzinfo or UTC)
    clauses = [
        ("trace.timestamp_ms", ">=", round_ms(since)),
        ("tag.status", "IS NOT NULL", None),
        ("tag.judge_score", "IS NULL", None),
        ("tag.judge_status", "IS NULL", None),
    ]
    experiment_id = traces_experiment_id(state, row.name)
    infos, _ = registry.search_traces(
        experiment_id, clauses, "timestamp_ms ASC", config.JUDGING_BATCH
    )

    # 2. Every Nth by arrival scored, a failed answer skipped, the rest passed over.
    count = row.judge_count
    for info in infos:
        selected = (count + 1) % row.judge_every == 0
        if selected and judged_this_hour(state, row.name, now) >= row.judge_max_per_hour:
            break
        count += 1
        if not selected:
            registry.set_trace_tag(info["trace_id"], "judge_status", "unselected")
        elif info.get("tags", {}).get("status") == "error":
            registry.set_trace_tag(info["trace_id"], "judge_status", "skipped")
        else:
            judge_trace(state, row.name, info, rubric)
            hour, judged = state.judgments[row.name]
            state.judgments[row.name] = (hour, judged + 1)

    # 3. The count kept, so a restart selects the same traces.
    with state.engine.begin() as connection:
        query = update(endpoint).where(endpoint.c.id == row.id).values(judge_count=count)
        connection.execute(query)


# Counted in memory per hour; after a restart the cap starts over, which is fine for a cap.
def judged_this_hour(state, name: str, now: datetime) -> int:
    hour = now.strftime("%Y-%m-%d %H")
    found, judged = state.judgments.get(name, (hour, 0))
    if found != hour:
        judged = 0
    state.judgments[name] = (hour, judged)
    return judged


def judge_trace(state, name: str, info: dict, rubric: str) -> None:
    """Asks the Judge to score the trace's answer and stores its Judge Score; an answer that
    fails or doesn't parse marks the trace judge-failed, never to be retried."""
    # 1. The Judge's answer to the rubric and the trace's content.
    registry = state.model_registry
    [trace] = registry.batch_get_traces([info["trace_id"]])
    request = {
        "messages": [{"role": "user", "content": judge_prompt(rubric, request_content(trace))}],
        "response_format": config.JUDGE_RESPONSE_FORMAT,
        "temperature": 0,
    }
    try:
        answer = state.cluster.ask_endpoint(
            config.JUDGE_ENDPOINT_NAME, request, config.JUDGING_TIMEOUT.total_seconds()
        )
        judgment = json.loads(answer["choices"][0]["message"]["content"])
        score, reason = int(judgment["score"]), str(judgment["reason"])
        if not config.JUDGE_SCORES[0] <= score <= config.JUDGE_SCORES[1]:
            raise ValueError(f"score {score} is out of range")
    except Exception:
        logger.warning("The Judge's score of trace %s failed", info["trace_id"], exc_info=True)
        registry.set_trace_tag(info["trace_id"], "judge_status", "failed")
        return

    # 2. The Judge Score on the trace, and in the Stats History at the trace's time.
    registry.set_feedback(info, "judge", score, "LLM_JUDGE", reason)
    registry.set_trace_tag(info["trace_id"], "judge_score", str(score))
    arrived = datetime.fromisoformat(info["request_time"])
    run_id = day_run_id(state, name, arrived)
    metric = {config.JUDGE_SCORE_METRIC: score}
    registry.log_metrics(run_id, metric, int(arrived.timestamp()), round_ms(arrived))


def judge_prompt(rubric: str, content: dict) -> str:
    conversation = "\n\n".join(
        f"{message.get('role')}: {text_of(message.get('content'))}"
        for message in content["messages"]
    )
    if content["tools"]:
        conversation += f"\n\nTools offered: {json.dumps(content['tools'])}"
    answer = content["answer"] or ""
    if content["tool_calls"]:
        answer += f"\n\nTool calls: {json.dumps(content['tool_calls'])}"
    return config.JUDGE_PROMPT.format(
        rubric=rubric,
        conversation=conversation,
        reasoning=content["reasoning"] or "(none)",
        answer=answer or "(empty)",
    )


def text_of(content) -> str:
    return content if isinstance(content, str) else json.dumps(content)
