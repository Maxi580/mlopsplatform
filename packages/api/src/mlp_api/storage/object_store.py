import os
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from mlp_core import config


class ObjectStore:
    """SeaweedFS: the platform bucket unless another is named; tests swap in a fake."""

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

    def upload_file(self, path: Path, key: str, bucket: str | None = None) -> None:
        self.client.upload_file(str(path), bucket or self.bucket, key)

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def download_url(self, key: str, bucket: str | None = None) -> str:
        return self.public_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket or self.bucket, "Key": key},
            ExpiresIn=int(config.DOWNLOAD_URL_LIFETIME.total_seconds()),
        )

    def read(self, bucket: str, key: str) -> bytes:
        return self.client.get_object(Bucket=bucket, Key=key)["Body"].read()

    # Uploads too large for one request go in parts, each PUT by the client to a presigned URL.
    def start_multipart_upload(self, bucket: str, key: str) -> str:
        return self.client.create_multipart_upload(Bucket=bucket, Key=key)["UploadId"]

    def part_upload_urls(self, bucket: str, key: str, upload_id: str, parts: int) -> list[str]:
        return [
            self.public_client.generate_presigned_url(
                "upload_part",
                Params={"Bucket": bucket, "Key": key, "UploadId": upload_id, "PartNumber": number},
                ExpiresIn=int(config.UPLOAD_URL_LIFETIME.total_seconds()),
            )
            for number in range(1, parts + 1)
        ]

    def complete_multipart_upload(self, bucket: str, key: str, upload_id: str) -> None:
        """Joins the parts the client uploaded, read from the store so it needn't send ETags."""
        pages = self.client.get_paginator("list_parts").paginate(
            Bucket=bucket, Key=key, UploadId=upload_id
        )
        parts = [
            {"PartNumber": part["PartNumber"], "ETag": part["ETag"]}
            for page in pages
            for part in page.get("Parts", [])
        ]
        try:
            self.client.complete_multipart_upload(
                Bucket=bucket, Key=key, UploadId=upload_id, MultipartUpload={"Parts": parts}
            )
        except ClientError as error:
            raise ValueError(f"{key} did not arrive completely: {error}") from None

    def size_of(self, bucket: str, prefix: str) -> int:
        """The bytes of every object whose key starts with the prefix."""
        return sum(item["Size"] for item in self.list_objects(bucket, prefix))

    def delete_all(self, bucket: str, prefix: str) -> None:
        # An empty prefix matches, and so would delete, the whole bucket.
        if not prefix.strip("/"):
            raise ValueError(f"Refusing to delete everything in bucket {bucket}")
        for item in self.list_objects(bucket, prefix):
            self.client.delete_object(Bucket=bucket, Key=item["Key"])

    def bucket_sizes(self) -> dict[str, int]:
        buckets = self.client.list_buckets()["Buckets"]
        return {bucket["Name"]: self.size_of(bucket["Name"], "") for bucket in buckets}

    def list_objects(self, bucket: str, prefix: str):
        pages = self.client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
        for page in pages:
            yield from page.get("Contents", [])
