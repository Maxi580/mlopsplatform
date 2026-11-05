from typing import Annotated

import typer
import yaml

from mlp_cli.api import api_client, exit_on_error
from mlp_core import api_paths, config

app = typer.Typer()


@app.callback(invoke_without_command=True)
def endpoints(context: typer.Context) -> None:
    """List Endpoints with model, status and URL; start, stop, restart, delete them or show keys."""
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
    widths = [max(len(row[column]) for row in [header, *rows]) for column in range(3)]
    for *aligned, endpoint_url in [header, *rows]:
        columns = [value.ljust(width) for value, width in zip(aligned, widths, strict=True)]
        typer.echo("  ".join([*columns, endpoint_url]))

    # 3. Why any of them failed.
    for e in listing:
        if e.get("failure"):
            typer.echo(f"{e['name']} failed: {e['failure']}")


@app.command()
def start(
    model: Annotated[str, typer.Argument(help="hf:org/name[@revision] or model:name[@version]")],
    name: Annotated[str, typer.Option(help="Names the Endpoint and its URL")],
    option: Annotated[
        list[str] | None,
        typer.Option(
            "--option",
            "-o",
            help="A serving option as key=value, e.g. max_model_len=8192. "
            f"gpu_memory_utilization: {config.VLLM_GPU_SHARE_HINT}",
        ),
    ] = None,
) -> None:
    """Start an Endpoint serving the model; it shows `waiting for GPU` while none is free."""
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
def restart(
    name: str,
    gpu_memory_utilization: Annotated[
        float | None,
        typer.Option(help=f"{config.VLLM_GPU_SHARE_HINT} {config.GPU_SHARE_LEFT_OUT_HINT}"),
    ] = None,
) -> None:
    """Restart the stopped or failed Endpoint at the same URL and Endpoint Key."""
    entered = gpu_memory_utilization is not None
    new = {"gpu_memory_utilization": gpu_memory_utilization} if entered else None
    with api_client() as client:
        response = client.post(api_paths.RESTART_ENDPOINT.format(name=name), json=new)
    restarted = exit_on_error(response).json()
    typer.echo(f"Restarted Endpoint {restarted['name']} ({restarted['status']})")
    typer.echo(f"OpenAI base URL: {restarted['url']}")


@app.command()
def delete(name: str) -> None:
    """Delete the stopped Endpoint from the list."""
    with api_client() as client:
        exit_on_error(client.delete(api_paths.ENDPOINT.format(name=name)))
    typer.echo(f"Deleted {name}")


@app.command()
def key(name: str) -> None:
    """Print the Endpoint's Endpoint Key, the bearer token its URL takes."""
    with api_client() as client:
        listing = exit_on_error(client.get(api_paths.ENDPOINTS)).json()
    # Newest first, so the first is the one older databases restart.
    named = [e for e in listing if e["name"] == name]
    if not named:
        typer.echo(f"No Endpoint {name}", err=True)
        raise typer.Exit(1)
    typer.echo(named[0]["key"])


@app.command("refresh-key")
def refresh_key(name: str) -> None:
    """Replace the Endpoint's Endpoint Key; the old one stops working at once."""
    with api_client() as client:
        path = api_paths.REFRESH_ENDPOINT_KEY.format(name=name)
        refreshed = exit_on_error(client.post(path)).json()
    typer.echo(f"New Endpoint Key of {name}: {refreshed['key']}")
