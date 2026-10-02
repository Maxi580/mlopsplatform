from dataclasses import dataclass
from pathlib import Path

import httpx
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import HfHubHTTPError

from mlp_core import config
from mlp_core.pipeline_request.references import base_model_reference, split_base_model_reference


@dataclass(frozen=True)
class HubModel:
    commit: str
    needs_remote_code: bool


class HuggingFace:
    """Hugging Face Hub lookups and downloads; tests swap in a fake."""

    def find_model(self, repo: str, revision: str, token: str | None) -> HubModel | None:
        hub = HfApi(token=token or False)
        try:
            info = hub.model_info(repo, revision=revision)
            if info.gated:
                hub.auth_check(repo)
        except (HfHubHTTPError, httpx.HTTPError):
            return None
        return HubModel(commit=info.sha, needs_remote_code="auto_map" in (info.config or {}))

    def download_model(self, repo: str, commit: str, directory: Path) -> None:
        snapshot_download(repo, revision=commit, local_dir=directory, token=False)


def pin_base_model(hugging_face: HuggingFace, reference: str, token: str | None) -> str:
    """The `hf:` Reference pinned to a commit; ValueError saying why it can't be."""
    repo, revision = split_base_model_reference(reference)
    model = hugging_face.find_model(repo, revision or config.DEFAULT_HF_REVISION, token)
    if model is None:
        reason = "is missing, gated for this token, or has no such revision"
        raise ValueError(f"{repo} on Hugging Face {reason}")
    if model.needs_remote_code:
        raise ValueError(f"{repo} needs remote code, which never runs here")
    return base_model_reference(repo, model.commit)
