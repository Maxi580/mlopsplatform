import ssl
from pathlib import Path

import httpx2 as httpx

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
    platform_ca = ssl.create_default_context(cafile=Path(profile["ca_cert"]).expanduser())
    path = token_file()
    headers = {"Authorization": f"Bearer {path.read_text()}"} if path.exists() else {}
    return httpx.Client(base_url=profile["url"], verify=platform_ca, headers=headers)
