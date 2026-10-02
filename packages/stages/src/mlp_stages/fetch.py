import os

from huggingface_hub import snapshot_download

from mlp_core import config
from mlp_core.pipeline_request.references import split_base_model_reference


def fetch(base_model: str) -> None:
    """Downloads the Base Model at its pinned commit into the Model Cache (HF_HOME)."""
    repo, commit = split_base_model_reference(base_model)
    token = os.environ.get(config.SECRET_ENV_VARS["hf_token"]) or None
    snapshot_download(repo, revision=commit, token=token)
