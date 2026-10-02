import os

from huggingface_hub import HfApi, scan_cache_dir, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.errors import CacheNotFound

from mlp_core import config
from mlp_core.pipeline_request.references import split_base_model_reference
from mlp_core.settings import size_in_bytes


def fetch(base_model: str, model_cache_size: str) -> None:
    """Downloads the Base Model at its pinned commit into the Model Cache (HF_HOME), if it fits."""
    repo, commit = split_base_model_reference(base_model)
    token = os.environ.get(config.SECRET_ENV_VARS["hf_token"]) or None

    # 1. The whole download fits beside what is cached; the API already evicted what it could.
    info = HfApi(token=token or False).model_info(repo, revision=commit, files_metadata=True)
    download = sum(file.size or 0 for file in info.siblings or [])
    cached, cached_revision = cached_sizes(repo, commit)
    needed = download - cached_revision
    capacity = size_in_bytes(model_cache_size)
    if cached + needed > capacity:
        raise RuntimeError(
            f"{base_model} needs {needed} bytes but the Model Cache has {capacity - cached} of "
            f"{capacity} free; free Base Models no Pipeline uses with `mlp cache free`, "
            "or raise model_cache_size"
        )

    # 2. The download.
    snapshot_download(repo, revision=commit, token=token)


def cached_sizes(repo: str, commit: str) -> tuple[int, int]:
    """The bytes the whole Hugging Face cache holds, and those of the repo at the commit."""
    try:
        cache = scan_cache_dir(HF_HUB_CACHE)
    except CacheNotFound:
        return 0, 0
    revision = sum(
        revision.size_on_disk
        for found in cache.repos
        if found.repo_id == repo
        for revision in found.revisions
        if revision.commit_hash == commit
    )
    return cache.size_on_disk, revision
