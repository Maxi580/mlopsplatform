from dataclasses import dataclass
from pathlib import Path

import httpx
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import HfHubHTTPError

from mlp_core import config
from mlp_core.pipeline_request.references import base_model_reference, split_base_model_reference


@dataclass(frozen=True)
class HubModel:
    commit: str
    needs_remote_code: bool
    # From its config.json; picks the tool-call parser an Endpoint serves it with.
    model_type: str | None = None


class HuggingFace:
    """Hugging Face Hub lookups and downloads; tests swap in a fake."""

    def __init__(self):
        # (repo, commit) -> bytes, from validation's lookup, so the download preview needn't ask.
        self.sizes: dict[tuple[str, str], int] = {}

    def find_model(self, repo: str, revision: str, token: str | None) -> HubModel | None:
        hub = HfApi(token=token or False)
        try:
            info = hub.model_info(repo, revision=revision, files_metadata=True)
            if info.gated:
                hub.auth_check(repo)
        except (HfHubHTTPError, httpx.HTTPError):
            return None
        self.sizes[(repo, info.sha)] = sum(file.size or 0 for file in info.siblings or [])
        model_config = info.config or {}
        return HubModel(
            commit=info.sha,
            needs_remote_code="auto_map" in model_config,
            model_type=model_config.get("model_type"),
        )

    def model_size(self, repo: str, commit: str, token: str | None) -> int | None:
        """The bytes a download of every file at the commit takes, or None if Hugging Face fails."""
        if (repo, commit) in self.sizes:
            return self.sizes[(repo, commit)]
        try:
            info = HfApi(token=token or False).model_info(
                repo, revision=commit, files_metadata=True
            )
        except (HfHubHTTPError, httpx.HTTPError):
            return None
        return sum(file.size or 0 for file in info.siblings or [])

    def model_file(self, repo: str, commit: str, path: str, token: str | None) -> bytes | None:
        """One file of the model at the commit, or None if it has none or Hugging Face fails."""
        try:
            return Path(
                hf_hub_download(repo, path, revision=commit, token=token or False)
            ).read_bytes()
        except (HfHubHTTPError, httpx.HTTPError):
            return None

    def dataset_file(self, repo: str, commit: str, path: str) -> bytes:
        return Path(
            hf_hub_download(repo, path, repo_type="dataset", revision=commit, token=False)
        ).read_bytes()

    def download_model(self, repo: str, commit: str, directory: Path) -> None:
        snapshot_download(repo, revision=commit, local_dir=directory, token=False)

    def search_models(self, word: str, token: str | None) -> list[dict]:
        """Text-generation models whose name holds the word, most downloaded first, each with
        its parameters and whether it is gated; none if Hugging Face fails."""
        hub = HfApi(token=token or False)
        try:
            found = hub.list_models(
                search=word,
                pipeline_tag="text-generation",
                sort="downloads",
                limit=config.BASE_MODEL_SEARCH_SCAN,
                expand=["safetensors", "gated"],
            )
            return [
                {
                    "name": model.id,
                    "parameters": model.safetensors.total if model.safetensors else None,
                    "gated": bool(model.gated),
                }
                for model in found
            ]
        except (HfHubHTTPError, httpx.HTTPError):
            return []


def pin_base_model(hugging_face: HuggingFace, reference: str, token: str | None) -> str:
    """The `hf:` Reference pinned to a commit; ValueError saying why it can't be."""
    return find_base_model(hugging_face, reference, token)[0]


def find_base_model(
    hugging_face: HuggingFace, reference: str, token: str | None
) -> tuple[str, HubModel]:
    """The `hf:` Reference pinned to a commit, and the model; ValueError saying why it can't be."""
    repo, revision = split_base_model_reference(reference)
    model = hugging_face.find_model(repo, revision or config.DEFAULT_HF_REVISION, token)
    if model is None:
        reason = "is missing, gated for this token, or has no such revision"
        raise ValueError(f"{repo} on Hugging Face {reason}")
    if model.needs_remote_code:
        raise ValueError(f"{repo} needs remote code, which never runs here")
    return base_model_reference(repo, model.commit), model


def search_base_models(hugging_face: HuggingFace, search: str, token: str | None) -> list[dict]:
    """The curated Base Models while `search` is empty, else the Hub's text-generation models
    whose name holds every word of it, most downloaded first; each with its Reference."""
    words = search.lower().split()
    if words:
        # The Hub matches one word, so the longest asks for the fewest others to drop.
        found = hugging_face.search_models(max(words, key=len), token)
        models = [m for m in found if all(word in m["name"].lower() for word in words)]
    else:
        models = config.CURATED_BASE_MODELS
    return [
        {**model, "reference": f"hf:{model['name']}"}
        for model in models[: config.BASE_MODEL_SEARCH_RESULTS]
    ]
