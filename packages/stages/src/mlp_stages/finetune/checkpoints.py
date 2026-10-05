import os
import time
from pathlib import Path

import boto3
from transformers import TrainerCallback

from mlp_core import config


class CheckpointUploads(TrainerCallback):
    """Saves a Checkpoint every `minutes` and uploads it under `prefix`, keeping only the newest."""

    def __init__(self, prefix: str, minutes: int, stop_after_checkpoint: bool):
        self.prefix = prefix
        self.interval = minutes * 60
        # The Smoke Test's resume case stops here, as a cancel would.
        self.stop_after_checkpoint = stop_after_checkpoint
        self.last_saved = time.monotonic()

    # The trainer saves only when told to, as the platform sets `save_strategy: no`.
    def on_step_end(self, args, state, control, **kwargs):
        if time.monotonic() - self.last_saved >= self.interval:
            control.should_save = True

    def on_save(self, args, state, control, **kwargs):
        upload_checkpoint(Path(args.output_dir) / f"checkpoint-{state.global_step}", self.prefix)
        self.last_saved = time.monotonic()
        if self.stop_after_checkpoint:
            control.should_training_stop = True


def upload_checkpoint(directory: Path, prefix: str) -> None:
    """Uploads the Checkpoint beside the one before it, then deletes that one."""
    store, bucket = object_store(), os.environ["S3_BUCKET"]
    uploaded = f"{prefix}{directory.name}/"
    # The completion marker goes last, so a Checkpoint holding it arrived whole.
    files = sorted(
        (path for path in directory.rglob("*") if path.is_file()),
        key=lambda path: path.name == config.CHECKPOINT_COMPLETE_FILE,
    )
    for path in files:
        store.upload_file(str(path), bucket, uploaded + path.relative_to(directory).as_posix())
    for key in checkpoint_keys(prefix):
        if not key.startswith(uploaded):
            store.delete_object(Bucket=bucket, Key=key)


def download_checkpoint(prefix: str, directory: Path) -> Path:
    """Where the newest whole Checkpoint under the prefix lies once downloaded."""
    keys = checkpoint_keys(prefix)
    whole = [
        key.removeprefix(prefix).split("/")[0]
        for key in keys
        if key.endswith(f"/{config.CHECKPOINT_COMPLETE_FILE}")
    ]
    if not whole:
        raise SystemExit(
            f"No whole Checkpoint at {prefix} to resume from; it may have been deleted"
        )
    newest = max(whole, key=lambda name: int(name.removeprefix("checkpoint-")))
    store, bucket = object_store(), os.environ["S3_BUCKET"]
    for key in keys:
        if key.startswith(f"{prefix}{newest}/"):
            path = directory / key.removeprefix(prefix)
            path.parent.mkdir(parents=True, exist_ok=True)
            store.download_file(bucket, key, str(path))
    return directory / newest


def delete_checkpoints(prefix: str) -> None:
    store, bucket = object_store(), os.environ["S3_BUCKET"]
    for key in checkpoint_keys(prefix):
        store.delete_object(Bucket=bucket, Key=key)


def checkpoint_keys(prefix: str) -> list[str]:
    pages = (
        object_store()
        .get_paginator("list_objects_v2")
        .paginate(Bucket=os.environ["S3_BUCKET"], Prefix=prefix)
    )
    return [item["Key"] for page in pages for item in page.get("Contents", [])]


# With the object store keys KFP's own steps use.
def object_store():
    return boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"])
