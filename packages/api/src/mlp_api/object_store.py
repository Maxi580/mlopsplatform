import os
from pathlib import Path

import boto3
from botocore.config import Config

from mlp_core import config


class ObjectStore:
    """SeaweedFS: files in the platform bucket, sizes and deletes in all; tests swap in a fake."""

    def __init__(self):
        self.bucket = os.environ["S3_BUCKET"]
        s3_config = Config(signature_version="s3v4", s3={"addressing_style": "path"})
        self.client = boto3.client(
            "s3", endpoint_url=os.environ["S3_ENDPOINT_URL"], config=s3_config
        )
        # Download URLs are signed for the public host, where Traefik forwards them to SeaweedFS.
        self.public_client = boto3.client(
            "s3", endpoint_url=os.environ["S3_PUBLIC_URL"], config=s3_config
        )

    def upload_file(self, path: Path, key: str) -> None:
        self.client.upload_file(str(path), self.bucket, key)

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def download_url(self, key: str) -> str:
        return self.public_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=int(config.DOWNLOAD_URL_LIFETIME.total_seconds()),
        )

    def size_of(self, bucket: str, prefix: str) -> int:
        """The bytes of every object whose key starts with the prefix."""
        return sum(item["Size"] for item in self.objects(bucket, prefix))

    def delete_all(self, bucket: str, prefix: str) -> None:
        # An empty prefix matches, and so would delete, the whole bucket.
        if not prefix.strip("/"):
            raise ValueError(f"Refusing to delete everything in bucket {bucket}")
        for item in self.objects(bucket, prefix):
            self.client.delete_object(Bucket=bucket, Key=item["Key"])

    def bucket_sizes(self) -> dict[str, int]:
        buckets = self.client.list_buckets()["Buckets"]
        return {bucket["Name"]: self.size_of(bucket["Name"], "") for bucket in buckets}

    def objects(self, bucket: str, prefix: str):
        pages = self.client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
        for page in pages:
            yield from page.get("Contents", [])
