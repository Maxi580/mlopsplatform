import io
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from mlp_api.app import app
from mlp_api.auth.account import set_password
from mlp_api.pipelines.cluster import KubeflowRun
from mlp_api.pipelines.compiler import StepEnvironment
from mlp_core import api_paths, config

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
    monkeypatch.setenv("MLFLOW_URL", "http://mlflow.test")
    monkeypatch.setenv("MLFLOW_BUCKET", "mlflow")
    monkeypatch.setenv("KUBEFLOW_URL", "http://ml-pipeline.test:8888")
    monkeypatch.setenv("KUBEFLOW_NAMESPACE", "kubeflow")
    monkeypatch.setenv("STAGES_IMAGE", "mlp-stages:real")
    monkeypatch.setenv("TRAINER_HF_IMAGE", "mlp-trainer-hf:real")
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
    """Keeps objects in memory per bucket, `objects` being the platform bucket."""

    def __init__(self):
        self.objects = {}
        self.buckets = {"platform": self.objects, "mlflow": {}, "mlpipeline": {}}

    def upload_file(self, path, key):
        self.objects[key] = path.read_bytes()

    def delete(self, key):
        self.objects.pop(key, None)

    def download_url(self, key):
        return f"https://objects.test/{key}?signature=x"

    def size_of(self, bucket, prefix):
        return sum(len(v) for k, v in self.buckets[bucket].items() if k.startswith(prefix))

    def delete_all(self, bucket, prefix):
        objects = self.buckets[bucket]
        for key in [key for key in objects if key.startswith(prefix)]:
            del objects[key]

    def bucket_sizes(self):
        return {bucket: self.size_of(bucket, "") for bucket in self.buckets}


@pytest.fixture
def object_store(api):
    api.app.state.object_store = FakeObjectStore()
    return api.app.state.object_store


class FakeModelRegistry:
    """Holds Model Versions in memory; their files go in the fake object store's mlflow bucket."""

    artifact_bucket = "mlflow"

    def __init__(self):
        self.versions = []
        self.deleted_models = []

    def model_versions(self):
        return list(self.versions)

    def delete_model_version(self, name, version):
        self.versions = [v for v in self.versions if (v.name, v.version) != (name, version)]

    def delete_registered_model(self, name):
        self.versions = [v for v in self.versions if v.name != name]
        self.deleted_models.append(name)


@pytest.fixture
def model_registry(api):
    api.app.state.model_registry = FakeModelRegistry()
    return api.app.state.model_registry


class FakeCluster:
    """Records Secrets and Kubeflow runs; tests set run states and pods waiting for GPUs."""

    steps = StepEnvironment(
        stages_image="mlp-stages:test",
        trainer_images={"hf": "mlp-trainer-hf:test"},
        model_cache_pvc="model-cache",
        object_store_url="http://seaweedfs.test:8333",
        object_store_bucket="platform",
    )

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
def logged_in_api(api, hugging_face, object_store, model_registry, cluster):
    api.post(api_paths.LOGIN, json={"password": PASSWORD})
    return api
