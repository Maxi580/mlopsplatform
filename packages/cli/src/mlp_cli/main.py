from typing import Annotated

import typer
import yaml

from mlp_cli import datasets
from mlp_cli.api import api_client, save_token
from mlp_cli.profile import build_pipeline_request, load_profile
from mlp_core import api_paths

app = typer.Typer(no_args_is_help=True)
app.add_typer(datasets.app, name="datasets")


@app.callback()
def mlp() -> None:
    """Client for the MLOps platform API."""


@app.command()
def login(
    password: Annotated[
        str, typer.Option(envvar="MLP_PASSWORD", prompt="Shared password", hide_input=True)
    ],
) -> None:
    """Log in with the shared account; later commands use the stored token for 12 hours."""
    with api_client() as client:
        response = client.post(api_paths.LOGIN, json={"password": password})
    if response.is_error:
        typer.echo(f"Login failed ({response.status_code}): {response.text}", err=True)
        raise typer.Exit(1)
    save_token(response.json()["token"])
    typer.echo("Logged in")


@app.command()
def validate(
    finetune: Annotated[str, typer.Option(help="Phases from the CLI Profile, e.g. sft,dpo")] = "",
    name: Annotated[str | None, typer.Option(help="Pipeline name; default: the Profile's")] = None,
) -> None:
    """Check the Pipeline Request built from the CLI Profile, without running anything."""
    profile = load_profile()
    try:
        request = build_pipeline_request(profile, name, [p for p in finetune.split(",") if p])
    except ValueError as error:
        typer.echo(error, err=True)
        raise typer.Exit(1) from None
    submission = {"request": request, "secrets": profile.get("secrets", {})}
    with api_client() as client:
        response = client.post(api_paths.VALIDATE_PIPELINE, json=submission)
    if response.status_code == 422:
        for error in response.json()["detail"]:
            path = ".".join(str(part) for part in error["loc"])
            typer.echo(f"{path}: {error['msg']}", err=True)
        raise typer.Exit(1)
    if response.is_error:
        typer.echo(f"Validation failed ({response.status_code}): {response.text}", err=True)
        raise typer.Exit(1)
    typer.echo(yaml.safe_dump(response.json()["request"], sort_keys=False))
    typer.echo("Valid")
