import os
from pathlib import Path

import boto3

from mlp_core.pipeline_request.references import dataset_key, split_dataset_reference


def download_dataset_version(reference: str, directory: Path) -> Path:
    """Where the Dataset Version's JSONL file lies once downloaded from the platform bucket."""
    name, version = split_dataset_reference(reference)
    path = directory / f"{name}-{version}.jsonl"
    object_store = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"])
    object_store.download_file(os.environ["S3_BUCKET"], dataset_key(name, version), str(path))
    return path
