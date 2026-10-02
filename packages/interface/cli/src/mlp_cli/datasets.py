from pathlib import Path
from typing import Annotated

import typer

from mlp_cli.api import api_client, download_client, exit_on_error
from mlp_core import api_paths

app = typer.Typer()


@app.callback(invoke_without_command=True)
def datasets(context: typer.Context) -> None:
    """List Datasets with their versions and sizes, or manage them."""
    if context.invoked_subcommand:
        return
    with api_client() as client:
        response = exit_on_error(client.get(api_paths.DATASETS))
    if not response.json():
        typer.echo("No Datasets yet; add one with `mlp datasets upload`")
        return
    rows = [
        (f"{dataset['name']}@{v['version']}", f"{v['size_bytes']:,} bytes", v["row_format"])
        for dataset in response.json()
        for v in dataset["versions"]
    ]
    width = max([len("Dataset Version"), *(len(row[0]) for row in rows)])
    for version, size, row_format in [("Dataset Version", "Size", "Row format"), *rows]:
        typer.echo(f"{version:<{width}}  {size:>15}  {row_format}")


@app.command()
def upload(
    file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    name: Annotated[str, typer.Option(help="Dataset name; reusing one adds a version")],
) -> None:
    """Upload a JSONL file as the next version of a Dataset."""
    with api_client() as client, file.open("rb") as content:
        response = client.post(
            api_paths.DATASET_VERSIONS.format(name=name),
            content=content,
            headers={"Content-Type": "application/x-ndjson"},
            timeout=None,
        )
    new_version = exit_on_error(response).json()
    typer.echo(f"Uploaded {name}@{new_version['version']} ({new_version['row_format']} rows)")


@app.command()
def download(
    dataset_version: Annotated[str, typer.Argument(help="name@version")],
    output: Annotated[
        Path | None, typer.Option("-o", help="Default: <name>-<version>.jsonl")
    ] = None,
) -> None:
    """Download a Dataset Version straight from the object store."""
    # 1. The presigned URL, from the API.
    name, version = split_dataset_version(dataset_version)
    with api_client() as client:
        path = api_paths.DATASET_DOWNLOAD.format(name=name, version=version)
        url = exit_on_error(client.get(path)).json()["url"]
    # 2. The file, streamed straight from the object store.
    output = output or Path(f"{name}-{version}.jsonl")
    with download_client() as client, client.stream("GET", url) as response:
        if response.is_error:
            typer.echo(f"Download failed ({response.status_code})", err=True)
            raise typer.Exit(1)
        with output.open("wb") as file:
            for chunk in response.iter_bytes():
                file.write(chunk)
    typer.echo(f"Saved {name}@{version} to {output}")


@app.command()
def delete(dataset_version: Annotated[str, typer.Argument(help="name@version")]) -> None:
    """Delete a Dataset Version and its file."""
    name, version = split_dataset_version(dataset_version)
    with api_client() as client:
        exit_on_error(client.delete(api_paths.DATASET_VERSION.format(name=name, version=version)))
    typer.echo(f"Deleted {name}@{version}")


def split_dataset_version(dataset_version: str) -> tuple[str, int]:
    name, _, version = dataset_version.partition("@")
    if not version.isdigit():
        raise typer.BadParameter(f"expected name@version, e.g. {name}@1, got {dataset_version}")
    return name, int(version)
