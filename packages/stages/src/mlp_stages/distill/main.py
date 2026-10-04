import json
import os
import tempfile
from pathlib import Path

import mlflow

from mlp_core import api_paths, config
from mlp_core.pipeline_request.schema import PipelineRequest
from mlp_stages.dataset_versions import download_dataset_version
from mlp_stages.distill.replies import distilled_row, prompt_messages
from mlp_stages.distill.teacher import teacher_replies
from mlp_stages.platform_api import call_step_route


def distill(pipeline_id: str, request: str, teacher_url: str, gpus: str, dataset: str) -> None:
    """Registers the prompts with their well-formed Teacher replies as a Dataset Version."""
    # 1. The `distill` block of the resolved request the compiler passed in, and its prompts.
    resolved = PipelineRequest.model_validate_json(request)
    distill = resolved.distill
    run_id = os.environ["MLFLOW_RUN_ID"]
    with tempfile.TemporaryDirectory() as scratch:
        path = download_dataset_version(distill.dataset, Path(scratch))
        lines = path.read_text().splitlines()
        prompts = [prompt_messages(json.loads(line)["prompt"]) for line in lines]

        # 2. The Teacher's reply to each prompt.
        replies = teacher_replies(resolved, prompts, teacher_url, int(gpus), Path(scratch))

    # 3. A row per well-formed reply; the others are dropped, with why (#17).
    rows, dropped = [], []
    for prompt, reply in zip(prompts, replies, strict=True):
        try:
            rows.append(distilled_row(prompt, reply, distill))
        except ValueError as reason:
            dropped.append({"prompt": prompt, "reply": reply, "reason": str(reason)})

    # 4. Counts and some dropped replies in the step's Run; too many dropped fail the Stage.
    client = mlflow.MlflowClient()
    client.log_metric(run_id, "distill/kept", len(rows))
    client.log_metric(run_id, "distill/dropped", len(dropped))
    print(f"Kept {len(rows)} of {len(prompts)} replies")
    if dropped:
        client.log_dict(run_id, dropped[: config.DISTILL_DROPPED_SAMPLES], "dropped_replies.json")
        print(f"Dropped {len(dropped)}, e.g. because {dropped[0]['reason']}")
    if len(dropped) > config.DISTILL_MAX_DROPPED_FRACTION * len(prompts):
        share = f"{config.DISTILL_MAX_DROPPED_FRACTION:.0%}"
        raise SystemExit(
            f"The Teacher's replies were malformed too often: dropped {len(dropped)} of "
            f"{len(prompts)}, more than {share}; see dropped_replies.json in the Run"
        )

    # 5. The rows as a Dataset Version, which KFP hands later Stages as `@distill`.
    content = "".join(json.dumps(row) + "\n" for row in rows).encode()
    registered = call_step_route(api_paths.DISTILL_PIPELINE, pipeline_id, content)
    Path(dataset).parent.mkdir(parents=True, exist_ok=True)
    Path(dataset).write_text(registered["dataset"])
    print(f"Registered {registered['dataset']}")
