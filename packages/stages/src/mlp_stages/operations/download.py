import os
from pathlib import Path

import boto3
from botocore.config import Config


def download(*files: str) -> None:
    """Downloads each `s3://<bucket>/<prefix>=<directory>` pair's objects into the directory."""
    client = object_store_client()
    for pair in files:
        source, directory = pair.split("=", 1)
        bucket, prefix = source.removeprefix("s3://").split("/", 1)
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                path = Path(directory) / item["Key"].removeprefix(prefix)
                path.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(bucket, item["Key"], str(path))


def object_store_client():
    s3_config = Config(signature_version="s3v4", s3={"addressing_style": "path"})
    return boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"], config=s3_config)
