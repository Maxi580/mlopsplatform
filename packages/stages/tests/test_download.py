from mlp_stages.operations import download

PREFIX = "1/run-abc/artifacts/model/"


class FakeS3:
    """Holds the mlflow bucket's objects and lists them a page at a time, as S3 does."""

    objects = {
        "mlflow": {
            f"{PREFIX}config.json": b"{}",
            f"{PREFIX}nested/model.safetensors": b"weights",
            "1/run-other/artifacts/model/config.json": b"other",
        }
    }

    def get_paginator(self, operation):
        return self

    def paginate(self, Bucket, Prefix):
        for key in self.objects[Bucket]:
            if key.startswith(Prefix):
                yield {"Contents": [{"Key": key}]}

    def download_file(self, bucket, key, path):
        with open(path, "wb") as file:
            file.write(self.objects[bucket][key])


def test_download_copies_every_file_of_the_model_version_into_its_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(download, "object_store_client", FakeS3)

    download.download(f"s3://mlflow/{PREFIX}={tmp_path / 'weights'}")

    files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
    assert files == ["weights/config.json", "weights/nested/model.safetensors"]
    assert (tmp_path / "weights" / "nested" / "model.safetensors").read_bytes() == b"weights"
