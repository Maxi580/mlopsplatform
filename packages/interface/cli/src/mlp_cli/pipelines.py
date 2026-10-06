from typing import Annotated

import typer
import yaml

from mlp_cli.api import api_client, exit_on_error
from mlp_cli.profile import build_pipeline_request, load_profile
from mlp_cli.secrets import collect_secrets
from mlp_core import api_paths

FinetunePhases = Annotated[str, typer.Option(help="Phases from the CLI Profile, e.g. sft,dpo")]
PipelineName = Annotated[str | None, typer.Option(help="Pipeline name; default: the Profile's")]
DryRun = Annotated[bool, typer.Option(help="Only list what the Pipeline would download")]
Evaluate = Annotated[bool, typer.Option(help="Run the Profile's benchmarks on its model")]
Distill = Annotated[bool, typer.Option(help="First distill the Profile's prompts with its Teacher")]
Quantize = Annotated[
    bool, typer.Option(help="Quantize the model as the Profile's quantize block says")
]
Speculate = Annotated[
    bool, typer.Option(help="Train a Speculator as the Profile's speculate block says")
]
Sweep = Annotated[
    bool, typer.Option(help="First search hyperparameters as the Profile's sweep block says")
]


def validate(
    finetune: FinetunePhases = "",
    name: PipelineName = None,
    evaluate: Evaluate = False,
    distill: Distill = False,
    quantize: Quantize = False,
    speculate: Speculate = False,
    sweep: Sweep = False,
) -> None:
    """Check the Pipeline Request built from the CLI Profile, without running anything."""
    chosen = (finetune, name, evaluate, distill, quantize, speculate, sweep)
    resolved = send_pipeline_request(api_paths.VALIDATE_PIPELINE, *chosen)
    typer.echo(yaml.safe_dump(resolved["request"], sort_keys=False))
    typer.echo("Valid")


def run(
    finetune: FinetunePhases = "",
    name: PipelineName = None,
    evaluate: Evaluate = False,
    distill: Distill = False,
    quantize: Quantize = False,
    speculate: Speculate = False,
    sweep: Sweep = False,
    dry_run: DryRun = False,
) -> None:
    """Submit the Pipeline Request built from the CLI Profile; returns once it is queued."""
    chosen = (finetune, name, evaluate, distill, quantize, speculate, sweep)
    if dry_run:
        print_downloads(send_pipeline_request(api_paths.VALIDATE_PIPELINE, *chosen))
        return
    pipeline = send_pipeline_request(api_paths.PIPELINES, *chosen)
    typer.echo(f"Submitted Pipeline {pipeline['id']}; follow it with `mlp ls`")


def rerun(pipeline_id: Annotated[int, typer.Argument(help="ID from `mlp ls`")]) -> None:
    """Submit a Pipeline's resolved request again as a new Pipeline, with Secrets read afresh."""
    with api_client() as client:
        original = exit_on_error(client.get(api_paths.PIPELINE.format(id=pipeline_id))).json()
    pipeline = send_with_secrets(api_paths.PIPELINES, original["request"], load_profile())
    typer.echo(f"Submitted Pipeline {pipeline['id']}; follow it with `mlp ls`")


def resume(pipeline_id: Annotated[int, typer.Argument(help="ID from `mlp ls`")]) -> None:
    """Continue a failed or cancelled Pipeline as a new one, reusing what it finished."""
    submission = {"secrets": collect_secrets(load_profile())}
    with api_client() as client:
        response = client.post(api_paths.RESUME_PIPELINE.format(id=pipeline_id), json=submission)
    pipeline = exit_on_error(response).json()
    typer.echo(
        f"Submitted Pipeline {pipeline['id']}, which resumes Pipeline {pipeline_id}; "
        "follow it with `mlp ls`"
    )


def ls() -> None:
    """List Pipelines with their Owner, status, Stages and links."""
    profile = load_profile()
    with api_client() as client:
        pipelines = exit_on_error(client.get(api_paths.PIPELINES)).json()
    if not pipelines:
        typer.echo("No Pipelines yet; start one with `mlp run`")
        return
    rows = [("ID", "Pipeline", "Owner", "Status", "Stages")] + [
        (str(p["id"]), p["name"], p["owner"], p["status"], ",".join(p["stages"])) for p in pipelines
    ]
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    for row in rows:
        typer.echo("  ".join(value.ljust(width) for value, width in zip(row, widths, strict=True)))
    typer.echo()
    # The KFP UI link is relative to the platform; MLflow's comes from KFP whole.
    for p in pipelines:
        if p["kubeflow_run_url"]:
            typer.echo(f"{p['id']} Kubeflow: {profile['url']}{p['kubeflow_run_url']}")
        if p["mlflow_run_url"]:
            typer.echo(f"{p['id']} MLflow:   {p['mlflow_run_url']}")
        if p.get("sweep"):
            values = [
                f"{k}={v}" for block in p["sweep"]["parameters"].values() for k, v in block.items()
            ]
            typer.echo(
                f"{p['id']} Sweep:    {' '.join(values)} (objective {p['sweep']['objective']})"
            )


def cancel(pipeline_id: Annotated[int, typer.Argument(help="ID from `mlp ls`")]) -> None:
    """Cancel a Pipeline; its run stops and its Secrets are deleted."""
    with api_client() as client:
        exit_on_error(client.post(api_paths.CANCEL_PIPELINE.format(id=pipeline_id)))
    typer.echo(f"Pipeline {pipeline_id} cancelled")


def send_pipeline_request(
    path: str,
    finetune: str,
    name: str | None,
    evaluate: bool,
    distill: bool,
    quantize: bool,
    speculate: bool,
    sweep: bool,
) -> dict:
    """The API's answer to the Profile's Pipeline Request and Secrets; exits on a rejection."""
    # 1. The request, with only the named Stages and Phases.
    profile = load_profile()
    try:
        phases = [p for p in finetune.split(",") if p]
        request = build_pipeline_request(
            profile, name, phases, evaluate, distill, quantize, speculate, sweep
        )
    except ValueError as error:
        typer.echo(error, err=True)
        raise typer.Exit(1) from None

    # 2. The API's verdict, with the Secrets beside the request.
    return send_with_secrets(path, request, profile)


def send_with_secrets(path: str, request: dict, profile: dict) -> dict:
    """The API's answer to the request with the Secrets read now; exits on a rejection."""
    submission = {"request": request, "secrets": collect_secrets(profile)}
    with api_client() as client:
        response = client.post(path, json=submission)
    if response.status_code == 422:
        for error in response.json()["detail"]:
            loc = ".".join(str(part) for part in error["loc"])
            typer.echo(f"{loc}: {error['msg']}", err=True)
        raise typer.Exit(1)
    return exit_on_error(response).json()


def print_downloads(answer: dict) -> None:
    for download in answer["downloads"]:
        size = "unknown size" if download["bytes"] is None else format_bytes(download["bytes"])
        cached = "  cached" if download["cached"] else ""
        typer.echo(f"{download['kind']}  {download['ref']}  {size}{cached}")
    typer.echo(
        f"{format_bytes(answer['download_bytes'])} to download, "
        f"{format_bytes(answer['cached_bytes'])} already cached"
    )


# Binary units, like the Web UI.
def format_bytes(size: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    index = 0
    while size >= 1024 and index < len(units) - 1:
        size /= 1024
        index += 1
    return f"{size} B" if index == 0 else f"{size:.1f} {units[index]}"
