from typing import Annotated

import typer

from mlp_cli import cache, datasets, models, pipelines
from mlp_cli.api import api_client, save_token
from mlp_cli.smoke_test import smoke_test
from mlp_core import api_paths

app = typer.Typer(no_args_is_help=True)
app.add_typer(datasets.app, name="datasets")
app.add_typer(models.app, name="models")
app.add_typer(cache.app, name="cache")
for command in (pipelines.validate, pipelines.run, pipelines.rerun, pipelines.ls, pipelines.cancel):
    app.command()(command)
app.command("smoke-test")(smoke_test)


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
