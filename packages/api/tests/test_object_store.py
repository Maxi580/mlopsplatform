from urllib.parse import parse_qs, urlsplit

import pytest

from mlp_api.storage.object_store import ObjectStore


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("S3_BUCKET", "platform")
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://seaweedfs.test:8333")
    return ObjectStore()


def signature(url: str) -> str:
    return parse_qs(urlsplit(url).query)["X-Amz-Signature"][0]


def test_a_download_url_is_signed_for_the_address_the_client_used(store):
    tunnel = store.download_url("datasets/chat", "localhost:8443")
    domain = store.download_url("datasets/chat", "platform.test")

    assert tunnel.startswith("https://localhost:8443/platform/datasets/chat?")
    assert domain.startswith("https://platform.test/platform/datasets/chat?")
    assert signature(tunnel) != signature(domain)


def test_part_upload_urls_are_signed_for_the_address_the_client_used(store):
    urls = store.part_upload_urls("mlflow", "model.bin", "upload-1", 2, "10.0.0.5:8443")

    assert [urlsplit(url).netloc for url in urls] == ["10.0.0.5:8443"] * 2
    assert parse_qs(urlsplit(urls[1]).query)["partNumber"] == ["2"]
