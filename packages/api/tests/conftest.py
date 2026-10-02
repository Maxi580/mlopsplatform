import io

import pytest
from fastapi.testclient import TestClient

from mlp_api.app import app
from mlp_api.auth.account import set_password
from mlp_core import api_paths

PASSWORD = "correct horse battery staple"
JWT_SECRET = "test-jwt-secret-of-at-least-32-bytes"


@pytest.fixture
def platform_database(settings_configmap_env, tmp_path, monkeypatch):
    database = tmp_path / "platform.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database}")
    monkeypatch.setenv("JWT_SECRET", JWT_SECRET)
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://seaweedfs.test:8333")
    monkeypatch.setenv("S3_PUBLIC_URL", "https://platform.test")
    monkeypatch.setenv("S3_BUCKET", "platform")
    return database


@pytest.fixture
def run_set_password(platform_database, monkeypatch):
    def run(password: str) -> None:
        monkeypatch.setattr("sys.stdin", io.StringIO(password + "\n"))
        set_password()

    return run


@pytest.fixture
def api(run_set_password):
    run_set_password(PASSWORD)
    # https, because the session cookie is Secure.
    with TestClient(app, base_url="https://testserver") as client:
        yield client


class FakeHuggingFace:
    """Records lookups and answers them from the test's setup."""

    def __init__(self):
        self.models = {}
        self.lookups = []

    def find_model(self, repo, revision, token):
        self.lookups.append((repo, revision, token))
        return self.models.get(repo)


@pytest.fixture
def hugging_face(api):
    api.app.state.hugging_face = FakeHuggingFace()
    return api.app.state.hugging_face


class FakeObjectStore:
    """Keeps objects in memory; download URLs point at the key."""

    def __init__(self):
        self.objects = {}

    def upload_file(self, path, key):
        self.objects[key] = path.read_bytes()

    def delete(self, key):
        self.objects.pop(key, None)

    def download_url(self, key):
        return f"https://objects.test/{key}?signature=x"


@pytest.fixture
def object_store(api):
    api.app.state.object_store = FakeObjectStore()
    return api.app.state.object_store


@pytest.fixture
def logged_in_api(api, hugging_face, object_store):
    api.post(api_paths.LOGIN, json={"password": PASSWORD})
    return api
