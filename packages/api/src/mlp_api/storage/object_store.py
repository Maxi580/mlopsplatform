import os
from functools import lru_cache
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from mlp_core import config


class ObjectStore:
    """SeaweedFS: the platform bucket unless another is named; tests swap in a fake."""

    def __init__(self):
        self.bucket = os.environ["S3_BUCKET"]
        self.client = s3_client(os.environ["S3_ENDPOINT_URL"])

    def upload_file(self, path: Path, key: str, bucket: str | None = None) -> None:
        self.client.upload_file(str(path), bucket or self.bucket, key)

    def write(self, key: str, content: bytes, bucket: str | None = None) -> None:
        self.client.put_object(Bucket=bucket or self.bucket, Key=key, Body=content)

    def delete(self, key: str, bucket: str | None = None) -> None:
        self.client.delete_object(Bucket=bucket or self.bucket, Key=key)

    def download_url(self, key: str, address: str, bucket: str | None = None) -> str:
        return public_client(address).generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket or self.bucket, "Key": key},
            ExpiresIn=int(config.DOWNLOAD_URL_LIFETIME.total_seconds()),
        )

    def read(self, bucket: str, key: str) -> bytes:
        return self.client.get_object(Bucket=bucket, Key=key)["Body"].read()

    # Uploads too large for one request go in parts, each PUT by the client to a presigned URL.
    def start_multipart_upload(self, bucket: str, key: str) -> str:
        return self.client.create_multipart_upload(Bucket=bucket, Key=key)["UploadId"]

    def part_upload_urls(
        self, bucket: str, key: str, upload_id: str, parts: int, address: str
    ) -> list[str]:
        return [
            public_client(address).generate_presigned_url(
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


# A signature covers the host, so URLs are signed for the address the client called the API at.
@lru_cache(maxsize=config.PUBLIC_S3_CLIENTS_KEPT)
def public_client(address: str):
    return s3_client(f"https://{address}")


def s3_client(url: str):
    s3_config = Config(signature_version="s3v4", s3={"addressing_style": "path"})
    # A session of its own: routes create clients in threads, and the default isn't thread-safe.
    return boto3.session.Session().client("s3", endpoint_url=url, config=s3_config)
