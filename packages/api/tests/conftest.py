import io
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from mlp_api.app import app
from mlp_api.auth.account import set_password
from mlp_api.pipelines.cluster import KubeflowRun
from mlp_api.pipelines.hugging_face import HubModel
from mlp_core import api_paths, config

from .test_datasets import CHAT, jsonl, upload

PASSWORD = "correct horse battery staple"
JWT_SECRET = "test-jwt-secret-of-at-least-32-bytes"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
COMMIT = "7ae557604adf67be50417f59c2c2f167def9a775"


@pytest.fixture
def platform_database(settings_configmap_env, tmp_path, monkeypatch):
    database = tmp_path / "platform.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database}")
    monkeypatch.setenv("JWT_SECRET", JWT_SECRET)
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://seaweedfs.test:8333")
    monkeypatch.setenv("S3_PUBLIC_URL", "https://platform.test")
    monkeypatch.setenv("S3_BUCKET", "platform")
    monkeypatch.setenv("KUBEFLOW_URL", "http://ml-pipeline.test:8888")
    monkeypatch.setenv("KUBEFLOW_NAMESPACE", "kubeflow")
    monkeypatch.setenv("STAGES_IMAGE", "mlp-stages:real")
    monkeypatch.setenv("MODEL_CACHE_PVC", "model-cache")
    # Tests reconcile by hand, so the background loop never races them.
    monkeypatch.setattr(config, "RECONCILE_INTERVAL", timedelta(days=1))
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


class FakeCluster:
    """Records Secrets and Kubeflow runs; tests set run states and pods waiting for GPUs."""

    stages_image = "mlp-stages:test"
    model_cache_pvc = "model-cache"

    def __init__(self):
        self.secrets = {}
        self.secret_created = {}
        self.runs = {}
        self.terminated = []
        self.waiting_for_gpu = set()

    def create_secret(self, name, values):
        self.secrets[name] = values
        self.secret_created[name] = datetime.now(UTC)

    def delete_secret(self, name):
        self.secrets.pop(name, None)

    def secret_ages(self):
        return {name: datetime.now(UTC) - self.secret_created[name] for name in self.secrets}

    def submit_run(self, display_name, pipeline_spec):
        run_id = f"run-{len(self.runs) + 1}"
        self.runs[run_id] = KubeflowRun(state="PENDING", mlflow_run_url=None)
        self.submitted = {"display_name": display_name, "pipeline_spec": pipeline_spec}
        return run_id

    def find_run(self, run_id):
        return self.runs.get(run_id)

    def terminate_run(self, run_id):
        self.terminated.append(run_id)

    def is_waiting_for_gpu(self, run_id):
        return run_id in self.waiting_for_gpu


@pytest.fixture
def cluster(api):
    api.app.state.cluster = FakeCluster()
    return api.app.state.cluster


@pytest.fixture
def logged_in_api(api, hugging_face, object_store, cluster):
    api.post(api_paths.LOGIN, json={"password": PASSWORD})
    return api


@pytest.fixture
def submittable(logged_in_api, hugging_face):
    hugging_face.models[BASE_MODEL] = HubModel(commit=COMMIT, needs_remote_code=False)
    upload(logged_in_api, "chat", jsonl(CHAT))
