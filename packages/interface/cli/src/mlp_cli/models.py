from typing import Annotated

import typer

from mlp_cli.api import api_client, exit_on_error
from mlp_cli.versions import split_version
from mlp_core import api_paths

app = typer.Typer()


@app.callback(invoke_without_command=True)
def models(context: typer.Context) -> None:
    """List Registered Models with their versions, sizes and lineage, or manage them."""
    if context.invoked_subcommand:
        return
    # 1. Every Model Version, from the API.
    with api_client() as client:
        response = exit_on_error(client.get(api_paths.MODELS))
    if not response.json():
        typer.echo("No Registered Models yet; every finetune Phase registers one")
        return

    # 2. One row per version, under a header, in aligned columns.
    rows = [
        (
            f"{model['name']}@{v['version']}",
            f"{v['size_bytes']:,} bytes",
            v["tags"].get("weights", ""),
            v["tags"].get("base_model", ""),
            f"#{v['tags']['pipeline']}" if "pipeline" in v["tags"] else "",
        )
        for model in response.json()
        for v in model["versions"]
    ]
    header = ("Model Version", "Size", "Weights", "Base Model", "Pipeline")
    width = max(len(row[0]) for row in [header, *rows])
    for version, size, weights, base_model, pipeline in [header, *rows]:
        typer.echo(f"{version:<{width}}  {size:>15}  {weights:<8}  {base_model}  {pipeline}")


@app.command()
def delete(model_version: Annotated[str, typer.Argument(help="name@version")]) -> None:
    """Delete a Model Version and its files."""
    name, version = split_version(model_version)
    with api_client() as client:
        exit_on_error(client.delete(api_paths.MODEL_VERSION.format(name=name, version=version)))
    typer.echo(f"Deleted {name}@{version}")
