import logging
import shutil
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import scan_cache_dir
from huggingface_hub.errors import CacheNotFound
from sqlalchemy import Engine

from mlp_api.in_use.users import refuse_while_in_use, users_of
from mlp_core import config
from mlp_core.pipeline_request.references import (
    base_model_reference,
    benchmark_directory,
    benchmark_reference,
    split_base_model_reference,
)
from mlp_core.settings import Settings, size_in_bytes

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheEntry:
    """A Base Model revision or a benchmark's datasets in the Model Cache."""

    kind: str
    reference: str
    size_bytes: int
    last_used: datetime
    # A benchmark is complete once fetch marked it; a Base Model revision always is.
    complete: bool = True


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
    """Frees unused entries, least recently used first, until below the high-water mark."""
    capacity = size_in_bytes(settings.model_cache_size)
    cached = list_cache_entries(model_cache)
    # Room for a download counts as used, so it fits below the mark once it arrives.
    used = sum(entry.size_bytes for entry in cached) + room_for_bytes
    high_water_mark = capacity * settings.model_cache_high_water_mark
    for entry in sorted(cached, key=lambda entry: entry.last_used):
        if used <= high_water_mark:
            return
        if not users_of(engine, entry.reference):
            delete_entry(model_cache, entry)
            logger.info("Evicted %s from the Model Cache", entry.reference)
            used -= entry.size_bytes


def make_room_for_downloads(state, download_bytes: int) -> None:
    """Evicts so the downloads fit; fetch fails if they still don't."""
    if download_bytes:
        evict_least_recently_used(state.engine, state.model_cache, state.settings, download_bytes)


def free_cache_entry(engine: Engine, model_cache: Path, reference: str) -> None:
    """Deletes the cached entry; LookupError if not cached, ValueError while in use."""
    found = find_cache_entry(model_cache, reference)
    if found is None:
        raise LookupError(f"{reference} is not in the Model Cache")
    refuse_while_in_use(engine, reference)
    delete_entry(model_cache, found)


def list_cache_entries(model_cache: Path) -> list[CacheEntry]:
    """Every Base Model revision and benchmark in the Model Cache, by Reference."""
    entries = cached_base_models(model_cache) + cached_benchmarks(model_cache)
    return sorted(entries, key=lambda entry: entry.reference)


def find_cache_entry(model_cache: Path, reference: str) -> CacheEntry | None:
    found = [entry for entry in list_cache_entries(model_cache) if entry.reference == reference]
    return found[0] if found else None


def cached_base_models(model_cache: Path) -> list[CacheEntry]:
    try:
        repos = scan_cache_dir(model_cache / config.HUB_DIRECTORY).repos
    except CacheNotFound:
        return []
    return [
        CacheEntry(
            kind="base_model",
            reference=base_model_reference(repo.repo_id, revision.commit_hash),
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
    ]


# Each lies in `benchmarks/<harness>/<task>/`, read by `evaluate` like weights by vLLM.
def cached_benchmarks(model_cache: Path) -> list[CacheEntry]:
    entries = []
    for directory in (model_cache / config.BENCHMARKS_DIRECTORY).glob("*/*"):
        files = [path.stat() for path in directory.rglob("*") if path.is_file()]
        entries.append(
            CacheEntry(
                kind="benchmark",
                reference=benchmark_reference(directory.parent.name, directory.name),
                size_bytes=sum(file.st_size for file in files),
                last_used=datetime.fromtimestamp(
                    max((file.st_atime for file in files), default=0), UTC
                ),
                complete=(directory / config.BENCHMARK_FETCHED_MARKER).exists(),
            )
        )
    return entries


# A Base Model also loses blobs no other revision shares, and its repo once its last revision goes.
def delete_entry(model_cache: Path, entry: CacheEntry) -> None:
    if entry.kind == "benchmark":
        shutil.rmtree(benchmark_directory(model_cache, entry.reference))
        return
    commit = split_base_model_reference(entry.reference)[1]
    scan_cache_dir(model_cache / config.HUB_DIRECTORY).delete_revisions(commit).execute()
