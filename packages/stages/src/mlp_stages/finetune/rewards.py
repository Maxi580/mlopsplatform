import json
import sys

from mlp_core import config
from mlp_core.pipeline_request.schema import Reward
from mlp_stages.sandbox import run_in_sandbox


def reward_functions(rewards: dict[str, Reward]) -> list:
    """One TRL reward function per reward, named after it, which scores in the Sandbox (#19)."""
    return [reward_function(name, reward.source) for name, reward in rewards.items()]


def reward_function(name: str, source: str):
    code = config.REWARD_SNIPPET.format(source=source)

    def score(
        prompts, completions, completion_ids, trainer_state, log_extra, log_metric, **columns
    ) -> list[float | None]:
        """Each completion's reward; one that fails scores 0.0 and is counted and reported."""
        # 1. The whole batch at once, each completion with its Dataset row.
        snippets = [
            {"code": code, "input": json.dumps({"sample": sample(completion), "item": item})}
            for completion, item in zip(completions, dataset_rows(prompts, columns), strict=True)
        ]
        results = run_in_sandbox(snippets)

        # 2. The scores, and why rewards failed, in the step's log and the Run's metrics.
        scores, failures = [], []
        for result in results:
            reward, failure = read_score(result)
            scores.append(reward)
            if failure:
                failures.append(failure)
        log_metric(f"rewards/{name}/errors", len(failures))
        if failures:
            print(
                f"Reward `{name}` failed on {len(failures)} of {len(results)} completions; "
                f"the first: {failures[0]}",
                file=sys.stderr,
            )
        return scores

    score.__name__ = name
    return score


def sample(completion) -> dict:
    """What the model wrote: its reply's text and the tool calls in it."""
    if isinstance(completion, str):
        return {"output_text": completion, "output_tools": []}
    [reply] = completion
    return {
        "output_text": reply.get("content") or "",
        "output_tools": reply.get("tool_calls") or [],
    }


def dataset_rows(prompts: list, columns: dict[str, list]) -> list[dict]:
    return [
        {"prompt": prompt, **{column: values[index] for column, values in columns.items()}}
        for index, prompt in enumerate(prompts)
    ]


def read_score(result: dict) -> tuple[float | None, str | None]:
    """The snippet's score, or 0.0 and why it failed."""
    if result["status"] != "ok":
        reason = config.REWARD_FAILURES.get(result["status"])
        return 0.0, reason or result["stderr"].strip().rsplit("\n", 1)[-1]
    try:
        value = json.loads(result["stdout"].rstrip().rsplit("\n", 1)[-1])
    except json.JSONDecodeError:
        # The reward exited before it returned, e.g. through sys.exit().
        return 0.0, "exited without returning a score"
    if value is None or isinstance(value, int | float):
        return value, None
    return 0.0, f"returned {value!r}, not a number or None"
