import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import scan_cache_dir
from huggingface_hub.errors import CacheNotFound
from sqlalchemy import Engine

from mlp_api.pipelines.lifecycle import pipelines_using, refuse_while_in_use
from mlp_core import config
from mlp_core.pipeline_request.references import base_model_reference
from mlp_core.settings import Settings, size_in_bytes

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CachedBaseModel:
    reference: str
    commit: str
    size_bytes: int
    last_used: datetime


def evict_forever(state, stop: threading.Event) -> None:
    """Evicts every EVICTION_INTERVAL until `stop` is set; a failed round is retried."""
    while not stop.wait(config.EVICTION_INTERVAL.total_seconds()):
        try:
            evict_least_recently_used(state.engine, state.model_cache, state.settings)
        except Exception:
            logger.exception("Evicting from the Model Cache failed")


def evict_least_recently_used(
    engine: Engine, model_cache: Path, settings: Settings, room_for_bytes: int = 0
) -> None:
    """Frees unused Base Models, least recently used first, until below the high-water mark."""
    capacity = size_in_bytes(settings.model_cache_size)
    cached = list_cached_base_models(model_cache)
    # Room for a download counts as used, so it fits below the mark once it arrives.
    used = sum(entry.size_bytes for entry in cached) + room_for_bytes
    high_water_mark = capacity * settings.model_cache_high_water_mark
    for entry in sorted(cached, key=lambda entry: entry.last_used):
        if used <= high_water_mark:
            return
        if not pipelines_using(engine, entry.reference):
            delete_revision(model_cache, entry.commit)
            logger.info("Evicted %s from the Model Cache", entry.reference)
            used -= entry.size_bytes


def make_room_for_downloads(state, download_bytes: int) -> None:
    """Evicts so the downloads fit; fetch fails if they still don't."""
    if download_bytes:
        evict_least_recently_used(state.engine, state.model_cache, state.settings, download_bytes)


def free_cached_base_model(engine: Engine, model_cache: Path, reference: str) -> None:
    """Deletes the cached revision; LookupError if not cached, ValueError while in use."""
    found = find_cached_base_model(model_cache, reference)
    if found is None:
        raise LookupError(f"{reference} is not in the Model Cache")
    refuse_while_in_use(engine, reference)
    delete_revision(model_cache, found.commit)


def list_cached_base_models(model_cache: Path) -> list[CachedBaseModel]:
    """Every Base Model revision in the Model Cache, by Reference."""
    try:
        repos = scan_cache_dir(model_cache).repos
    except CacheNotFound:
        return []
    return sorted(
        (
            CachedBaseModel(
                reference=base_model_reference(repo.repo_id, revision.commit_hash),
                commit=revision.commit_hash,
                size_bytes=revision.size_on_disk,
                # Reading weights updates their access time (daily under relatime), which covers
                # Pipelines and Endpoints alike.
                last_used=datetime.fromtimestamp(
                    max((file.blob_last_accessed for file in revision.files), default=0), UTC
                ),
            )
            for repo in repos
            if repo.repo_type == "model"
            for revision in repo.revisions
        ),
        key=lambda entry: entry.reference,
    )


def find_cached_base_model(model_cache: Path, reference: str) -> CachedBaseModel | None:
    found = [
        entry for entry in list_cached_base_models(model_cache) if entry.reference == reference
    ]
    return found[0] if found else None


# Also deletes blobs no other revision shares, and the repo once its last revision goes.
def delete_revision(model_cache: Path, commit: str) -> None:
    scan_cache_dir(model_cache).delete_revisions(commit).execute()
