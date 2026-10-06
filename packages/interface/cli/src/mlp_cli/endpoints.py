from typing import Annotated

import typer
import yaml

from mlp_cli.api import api_client, exit_on_error
from mlp_core import api_paths

app = typer.Typer()


@app.callback(invoke_without_command=True)
def endpoints(context: typer.Context) -> None:
    """List Endpoints with model, status and URL; start, stop, delete them or show their keys."""
    if context.invoked_subcommand:
        return
    # 1. Every Endpoint, stopped ones included, from the API.
    with api_client() as client:
        listing = exit_on_error(client.get(api_paths.ENDPOINTS)).json()
    if not listing:
        typer.echo("No Endpoints yet; start one with `mlp endpoints start`")
        return

    # 2. One row per Endpoint, under a header, in aligned columns.
    rows = [(e["name"], e["model"], e["status"], e["url"]) for e in listing]
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
    typer.echo(f"OpenAI base URL: {started['url']}")


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


@app.command()
def key(name: str) -> None:
    """Print the running Endpoint's Endpoint Key, the bearer token its URL takes."""
    with api_client() as client:
        listing = exit_on_error(client.get(api_paths.ENDPOINTS)).json()
    # A stopped Endpoint keeps its row, and its name may be running again.
    running = [e for e in listing if e["name"] == name and e["status"] != "stopped"]
    if not running:
        typer.echo(f"No Endpoint {name} is running", err=True)
        raise typer.Exit(1)
    typer.echo(running[0]["key"])


@app.command("refresh-key")
def refresh_key(name: str) -> None:
    """Replace the Endpoint's Endpoint Key; the old one stops working at once."""
    with api_client() as client:
        path = api_paths.REFRESH_ENDPOINT_KEY.format(name=name)
        refreshed = exit_on_error(client.post(path)).json()
    typer.echo(f"New Endpoint Key of {name}: {refreshed['key']}")
