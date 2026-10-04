import os
from pathlib import Path

from huggingface_hub import HfApi, scan_cache_dir, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.errors import CacheNotFound

from mlp_core import config
from mlp_core.pipeline_request.references import benchmark_directory, split_base_model_reference
from mlp_core.settings import size_in_bytes
from mlp_stages.evaluate.harness import download_benchmark


def fetch(model_cache_size: str, references: str) -> None:
    """Downloads the pinned Base Models and the benchmarks' datasets into the Model Cache."""
    chosen = [reference for reference in references.split(",") if reference]
    base_models = [reference for reference in chosen if reference.startswith("hf:")]
    benchmarks = [b for b in chosen if b not in base_models and not marker(b).exists()]
    token = os.environ.get(config.SECRET_ENV_VARS["hf_token"]) or None

    # 1. The whole download fits beside what is cached; the API already evicted what it could.
    needed = sum(missing_bytes(reference, token) for reference in base_models)
    needed += sum(config.BENCHMARKS[benchmark]["size_bytes"] for benchmark in benchmarks)
    cached = cached_bytes()
    capacity = size_in_bytes(model_cache_size)
    if cached + needed > capacity:
        raise RuntimeError(
            f"{', '.join(chosen)} needs {needed} bytes but the Model Cache has "
            f"{capacity - cached} of {capacity} free; free what no Pipeline uses with "
            "`mlp cache free`, or raise model_cache_size"
        )

    # 2. The Base Models at their pinned commits.
    for reference in base_models:
        repo, commit = split_base_model_reference(reference)
        snapshot_download(repo, revision=commit, token=token)

    # 3. The benchmarks; one that fails to download is logged as NA by `evaluate`.
    for benchmark in benchmarks:
        if download_benchmark(benchmark):
            marker(benchmark).touch()
        else:
            print(f"Downloading {benchmark} failed; see its harness's log above")


# Exists once all of the benchmark's datasets are downloaded.
def marker(benchmark: str) -> Path:
    directory = benchmark_directory(Path(config.MODEL_CACHE_PATH), benchmark)
    return directory / config.BENCHMARK_FETCHED_MARKER


def missing_bytes(base_model: str, token: str | None) -> int:
    """The bytes of the Base Model's files at its commit that aren't cached yet."""
    repo, commit = split_base_model_reference(base_model)
    info = HfApi(token=token or False).model_info(repo, revision=commit, files_metadata=True)
    download = sum(file.size or 0 for file in info.siblings or [])
    try:
        repos = scan_cache_dir(HF_HUB_CACHE).repos
    except CacheNotFound:
        return download
    cached = sum(
        revision.size_on_disk
        for found in repos
        if found.repo_id == repo
        for revision in found.revisions
        if revision.commit_hash == commit
    )
    return download - cached


def cached_bytes() -> int:
    """What the Model Cache holds: the Hugging Face cache and the benchmarks."""
    try:
        hub = scan_cache_dir(HF_HUB_CACHE).size_on_disk
    except CacheNotFound:
        hub = 0
    benchmarks = Path(config.MODEL_CACHE_PATH) / config.BENCHMARKS_DIRECTORY
    return hub + sum(path.stat().st_size for path in benchmarks.rglob("*") if path.is_file())
