from typing import Annotated

import typer
import yaml

from mlp_cli.api import api_client, exit_on_error
from mlp_cli.profile import build_pipeline_request, load_profile
from mlp_cli.secrets import collect_secrets
from mlp_core import api_paths

FinetunePhases = Annotated[str, typer.Option(help="Phases from the CLI Profile, e.g. sft,dpo")]
PipelineName = Annotated[str | None, typer.Option(help="Pipeline name; default: the Profile's")]


def validate(finetune: FinetunePhases = "", name: PipelineName = None) -> None:
    """Check the Pipeline Request built from the CLI Profile, without running anything."""
    resolved = send_pipeline_request(api_paths.VALIDATE_PIPELINE, finetune, name)
    typer.echo(yaml.safe_dump(resolved["request"], sort_keys=False))
    typer.echo("Valid")


def run(finetune: FinetunePhases = "", name: PipelineName = None) -> None:
    """Submit the Pipeline Request built from the CLI Profile; returns once it is queued."""
    pipeline = send_pipeline_request(api_paths.PIPELINES, finetune, name)
    typer.echo(f"Submitted Pipeline {pipeline['id']}; follow it with `mlp ls`")


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


def cancel(pipeline_id: Annotated[int, typer.Argument(help="ID from `mlp ls`")]) -> None:
    """Cancel a Pipeline; its run stops and its Secrets are deleted."""
    with api_client() as client:
        exit_on_error(client.post(api_paths.CANCEL_PIPELINE.format(id=pipeline_id)))
    typer.echo(f"Pipeline {pipeline_id} cancelled")


def send_pipeline_request(path: str, finetune: str, name: str | None) -> dict:
    """The API's answer to the Profile's Pipeline Request and Secrets; exits on a rejection."""
    # 1. The request, with only the named Stages and Phases.
    profile = load_profile()
    try:
        request = build_pipeline_request(profile, name, [p for p in finetune.split(",") if p])
    except ValueError as error:
        typer.echo(error, err=True)
        raise typer.Exit(1) from None

    # 2. The API's verdict, with the Secrets beside the request.
    submission = {"request": request, "secrets": collect_secrets(profile)}
    with api_client() as client:
        response = client.post(path, json=submission)
    if response.status_code == 422:
        for error in response.json()["detail"]:
            loc = ".".join(str(part) for part in error["loc"])
            typer.echo(f"{loc}: {error['msg']}", err=True)
        raise typer.Exit(1)
    return exit_on_error(response).json()
