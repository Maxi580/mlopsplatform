from typing import Annotated

import typer
import yaml

from mlp_cli.api import api_client, exit_on_error
from mlp_cli.profile import load_profile
from mlp_core import api_paths

app = typer.Typer()


@app.callback(invoke_without_command=True)
def endpoints(context: typer.Context) -> None:
    """List Endpoints with their model, status and URL, or start, stop and delete them."""
    if context.invoked_subcommand:
        return
    # 1. Every Endpoint, stopped ones included, from the API.
    with api_client() as client:
        listing = exit_on_error(client.get(api_paths.ENDPOINTS)).json()
    if not listing:
        typer.echo("No Endpoints yet; start one with `mlp endpoints start`")
        return

    # 2. One row per Endpoint, under a header, in aligned columns.
    url = load_profile()["url"]
    rows = [(e["name"], e["model"], e["status"], url + e["url"]) for e in listing]
    header = ("Endpoint", "Model", "Status", "URL")
    name_width = max(len(row[0]) for row in [header, *rows])
    model_width = max(len(row[1]) for row in [header, *rows])
    for name, model, status, endpoint_url in [header, *rows]:
        typer.echo(f"{name:<{name_width}}  {model:<{model_width}}  {status:<8}  {endpoint_url}")


@app.command()
def start(
    model: Annotated[str, typer.Argument(help="hf:org/name[@revision] or model:name[@version]")],
    name: Annotated[str, typer.Option(help="Names the Endpoint and its URL")],
    option: Annotated[
        list[str] | None,
        typer.Option(
            "--option", "-o", help="A serving option as key=value, e.g. max_model_len=8192"
        ),
    ] = None,
) -> None:
    """Start an Endpoint serving the model; it waits as pending while GPUs are busy."""
    # 1. The serving options, their values read as YAML so numbers and booleans keep their type.
    options = {}
    for pair in option or []:
        key, separator, value = pair.partition("=")
        if not separator:
            raise typer.BadParameter(f"{pair} is not key=value", param_hint="--option")
        options[key] = yaml.safe_load(value)

    # 2. The Endpoint, from the API.
    with api_client() as client:
        response = client.post(api_paths.ENDPOINTS, json={"name": name, "model": model, **options})
    started = exit_on_error(response).json()
    typer.echo(f"Started Endpoint {started['name']} ({started['status']})")
    typer.echo(f"OpenAI base URL: {load_profile()['url']}{started['url']}")


@app.command()
def stop(name: str) -> None:
    """Stop the Endpoint and free its GPUs."""
    with api_client() as client:
        exit_on_error(client.post(api_paths.STOP_ENDPOINT.format(name=name)))
    typer.echo(f"Stopped {name}")


@app.command()
def delete(name: str) -> None:
    """Delete the stopped Endpoint from the list."""
    with api_client() as client:
        exit_on_error(client.delete(api_paths.ENDPOINT.format(name=name)))
    typer.echo(f"Deleted {name}")
