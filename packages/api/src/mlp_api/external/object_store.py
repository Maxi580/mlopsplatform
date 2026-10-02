import os
from pathlib import Path

import boto3
from botocore.config import Config

from mlp_core import config


class ObjectStore:
    """The platform bucket in SeaweedFS; tests swap in a fake."""

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
