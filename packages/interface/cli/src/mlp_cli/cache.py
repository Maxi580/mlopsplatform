from datetime import datetime
from typing import Annotated

import typer

from mlp_cli.api import api_client, exit_on_error
from mlp_core import api_paths, config

app = typer.Typer()


@app.callback(invoke_without_command=True)
def cache(context: typer.Context) -> None:
    """List the Base Models and benchmarks in the Model Cache with sizes and last use."""
    if context.invoked_subcommand:
        return
    # 1. Everything cached, from the API.
    with api_client() as client:
        listing = exit_on_error(client.get(api_paths.MODEL_CACHE)).json()
    used = sum(entry["size_bytes"] for entry in listing["entries"])
    typer.echo(f"{used:,} of {listing['capacity_bytes']:,} bytes used")
    if not listing["entries"]:
        typer.echo("Nothing cached; fetch downloads Base Models and benchmarks once")
        return

    # 2. One row per entry, under a header, in aligned columns.
    rows = [
        (
            config.CACHE_ENTRY_KINDS[entry["kind"]],
            entry["reference"],
            f"{entry['size_bytes']:,} bytes",
            datetime.fromisoformat(entry["last_used"]).strftime("%Y-%m-%d %H:%M"),
        )
        for entry in listing["entries"]
    ]
    header = ("Kind", "Reference", "Size", "Last used")
    width = max(len(row[1]) for row in [header, *rows])
    for kind, reference, size, last_used in [header, *rows]:
        typer.echo(f"{kind:<10}  {reference:<{width}}  {size:>19}  {last_used}")


@app.command()
def free(
    reference: Annotated[str, typer.Argument(help="hf:org/name@commit or harness:task")],
) -> None:
    """Delete a cached Base Model or benchmark that no Pipeline uses."""
    with api_client() as client:
        exit_on_error(client.delete(api_paths.MODEL_CACHE, params={"reference": reference}))
    typer.echo(f"Freed {reference}")
