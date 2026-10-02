import typer


def split_version(name_version: str) -> tuple[str, int]:
    """The name and version of a `name@version` argument."""
    name, _, version = name_version.partition("@")
    if not version.isdigit():
        raise typer.BadParameter(f"expected name@version, e.g. {name}@1, got {name_version}")
    return name, int(version)
