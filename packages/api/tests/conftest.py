import io

import pytest
from fastapi.testclient import TestClient

from mlp_api.account import set_password
from mlp_api.app import app
from mlp_core import api_paths

PASSWORD = "correct horse battery staple"
JWT_SECRET = "test-jwt-secret-of-at-least-32-bytes"


@pytest.fixture
def platform_database(settings_configmap_env, tmp_path, monkeypatch):
    database = tmp_path / "platform.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database}")
    monkeypatch.setenv("JWT_SECRET", JWT_SECRET)
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


class FakeCluster:
    """Records what the API asks of the outside world and answers from the test's setup."""

    def __init__(self):
        self.hub_models = {}
        self.hub_lookups = []

    def find_hub_model(self, repo, revision, token):
        self.hub_lookups.append((repo, revision, token))
        return self.hub_models.get(repo)


@pytest.fixture
def cluster(api):
    api.app.state.cluster = FakeCluster()
    return api.app.state.cluster


@pytest.fixture
def logged_in_api(api, cluster):
    api.post(api_paths.LOGIN, json={"password": PASSWORD})
    return api
