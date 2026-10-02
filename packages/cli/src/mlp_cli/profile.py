from pathlib import Path

import yaml


def mlp_directory() -> Path:
    return Path.home() / ".mlp"


def load_profile() -> dict:
    """The CLI Profile; `url` names the platform and `ca_cert` the platform CA to trust."""
    return yaml.safe_load((mlp_directory() / "profile.yaml").read_text())
