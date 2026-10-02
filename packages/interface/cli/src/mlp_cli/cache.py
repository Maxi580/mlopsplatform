from datetime import datetime
from typing import Annotated

import typer

from mlp_cli.api import api_client, exit_on_error
from mlp_core import api_paths

app = typer.Typer()


@app.callback(invoke_without_command=True)
def cache(context: typer.Context) -> None:
    """List the Base Models in the Model Cache with their sizes and last use, or free them."""
    if context.invoked_subcommand:
        return
    # 1. Every cached Base Model, from the API.
    with api_client() as client:
        listing = exit_on_error(client.get(api_paths.CACHED_BASE_MODELS)).json()
    used = sum(entry["size_bytes"] for entry in listing["base_models"])
    typer.echo(f"{used:,} of {listing['capacity_bytes']:,} bytes used")
    if not listing["base_models"]:
        typer.echo("No cached Base Models; fetch downloads one per Pipeline")
        return

    # 2. One row per Base Model, under a header, in aligned columns.
    rows = [
        (
            entry["reference"],
            f"{entry['size_bytes']:,} bytes",
            datetime.fromisoformat(entry["last_used"]).strftime("%Y-%m-%d %H:%M"),
        )
        for entry in listing["base_models"]
    ]
    header = ("Base Model", "Size", "Last used")
    width = max(len(row[0]) for row in [header, *rows])
    for reference, size, last_used in [header, *rows]:
        typer.echo(f"{reference:<{width}}  {size:>19}  {last_used}")


@app.command()
def free(reference: Annotated[str, typer.Argument(help="hf:org/name@commit")]) -> None:
    """Delete a cached Base Model that no Pipeline uses."""
    with api_client() as client:
        exit_on_error(client.delete(api_paths.CACHED_BASE_MODELS, params={"reference": reference}))
    typer.echo(f"Freed {reference}")
