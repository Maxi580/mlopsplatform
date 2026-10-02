from pathlib import Path
from typing import Annotated

import typer

from mlp_cli.api import api_client, download_file, exit_on_error, object_store_client
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


@app.command()
def upload(
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    name: Annotated[str, typer.Option(help="Registered Model name; reusing one adds a version")],
    base: Annotated[
        str | None, typer.Option(help="An Adapter's base: hf:org/name or model:name@version")
    ] = None,
    tool_parser: Annotated[
        str | None, typer.Option(help="vLLM's tool-call parser, if the platform knows none")
    ] = None,
) -> None:
    """Upload a model directory (full weights, or an Adapter with --base) as a Model Version."""
    # 1. Every file but hidden ones such as .git, checked by the API before anything travels.
    files = {
        path.relative_to(directory).as_posix(): path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and not any(p.startswith(".") for p in path.relative_to(directory).parts)
    }
    listing = [{"path": path, "size_bytes": file.stat().st_size} for path, file in files.items()]
    with api_client() as client:
        started = exit_on_error(
            client.post(
                api_paths.MODEL_UPLOADS,
                json={"name": name, "files": listing, "base": base, "tool_parser": tool_parser},
            )
        ).json()

    # 2. Each file in parts, straight to the object store.
    size = started["part_size_bytes"]
    with object_store_client() as client:
        for file in started["files"]:
            typer.echo(f"Uploading {file['path']}")
            with files[file["path"]].open("rb") as content:
                for url in file["part_urls"]:
                    response = client.put(url, content=content.read(size))
                    if response.is_error:
                        typer.echo(f"Upload failed ({response.status_code})", err=True)
                        raise typer.Exit(1)

    # 3. The checks, then the new version.
    with api_client() as client:
        complete = api_paths.MODEL_UPLOAD_COMPLETE.format(id=started["id"])
        new_version = exit_on_error(client.post(complete, timeout=None)).json()
    typer.echo(f"Uploaded {new_version['name']}@{new_version['version']}")


@app.command()
def download(
    model_version: Annotated[str, typer.Argument(help="name@version")],
    output: Annotated[Path, typer.Option("-o", help="Directory to save the files in")],
) -> None:
    """Download every file of a Model Version straight from the object store."""
    name, version = split_version(model_version)
    with api_client() as client:
        path = api_paths.MODEL_VERSION_FILES.format(name=name, version=version)
        files = exit_on_error(client.get(path)).json()["files"]
    with object_store_client() as client:
        for file in files:
            download_file(client, file["url"], output / file["path"])
    typer.echo(f"Saved {name}@{version} to {output}")
