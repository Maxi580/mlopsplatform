import ssl
from pathlib import Path

import httpx2 as httpx
import typer

from mlp_cli.profile import load_profile, mlp_directory


def token_file() -> Path:
    return mlp_directory() / "token"


def save_token(token: str) -> None:
    path = token_file()
    path.touch(mode=0o600)
    path.write_text(token)


def api_client() -> httpx.Client:
    """A client for the platform API that trusts only the platform CA and sends the login token."""
    profile = load_profile()
    path = token_file()
    headers = {"Authorization": f"Bearer {path.read_text()}"} if path.exists() else {}
    return httpx.Client(base_url=profile["url"], verify=platform_ca(profile), headers=headers)


# Presigned URLs carry their own signature, which the login token would clash with.
def download_client() -> httpx.Client:
    return httpx.Client(verify=platform_ca(load_profile()), timeout=None)


def exit_on_error(response: httpx.Response) -> httpx.Response:
    """The response, unless it failed: then its reason is printed and the command exits."""
    if response.is_error:
        try:
            reason = response.json()["detail"]
        except (ValueError, KeyError, TypeError):
            reason = response.text
        typer.echo(f"Failed ({response.status_code}): {reason}", err=True)
        raise typer.Exit(1)
    return response


def platform_ca(profile: dict) -> ssl.SSLContext:
    return ssl.create_default_context(cafile=Path(profile["ca_cert"]).expanduser())
