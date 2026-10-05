import io
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from mlp_api.app import app
from mlp_api.auth.account import set_password
from mlp_api.endpoints.environment import EndpointEnvironment
from mlp_api.models.mlflow import ModelVersion
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
    monkeypatch.setenv("TRAINER_UNSLOTH_IMAGE", "mlp-trainer-unsloth:real")
    monkeypatch.setenv("MODEL_CACHE_PVC", "model-cache")
    monkeypatch.setenv("API_URL", "http://api.mlp.test:8000")
    monkeypatch.setenv("SANDBOX_URL", "http://sandbox.mlp.test:8090")
    monkeypatch.setenv("PLATFORM_NAMESPACE", "mlp")
    monkeypatch.setenv("DOMAIN", "platform.test")
    monkeypatch.setenv("VLLM_IMAGE", "vllm/vllm-openai:real")
    monkeypatch.setenv("MODEL_CACHE_HOST_PATH", "/var/lib/mlp/model-cache")
    # Tests reconcile by hand, so the background loop never races them.
    monkeypatch.setattr(config, "RECONCILE_INTERVAL", timedelta(days=1))
    monkeypatch.setattr(config, "EVICTION_INTERVAL", timedelta(days=1))
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
        # Repo -> its files, for downloads.
        self.files = {}
        # Repo -> the size of all its files.
        self.sizes = {}

    def find_model(self, repo, revision, token):
        self.lookups.append((repo, revision, token))
        return self.models.get(repo)

    def model_size(self, repo, commit, token):
        return self.sizes.get(repo)

    def model_file(self, repo, commit, path, token):
        return self.files.get(repo, {}).get(path)

    def download_model(self, repo, commit, directory):
        for path, content in self.files[repo].items():
            (directory / path).parent.mkdir(parents=True, exist_ok=True)
            (directory / path).write_bytes(content)


@pytest.fixture
def hugging_face(api):
    api.app.state.hugging_face = FakeHuggingFace()
    return api.app.state.hugging_face


class FakeObjectStore:
    """Keeps objects in memory per bucket, `objects` being the platform bucket.

    A multipart upload's parts arrive through `receive_part`, as a client's PUT to a part URL.
    """

    bucket = "platform"

    def __init__(self):
        self.objects = {}
        self.buckets = {"platform": self.objects, "mlflow": {}, "mlpipeline": {}}
        # Part URL -> its content, and multipart upload ID -> its parts' URLs.
        self.parts = {}
        self.multipart_uploads = {}

    def upload_file(self, path, key, bucket="platform"):
        self.buckets[bucket][key] = path.read_bytes()

    def delete(self, key):
        self.objects.pop(key, None)

    def download_url(self, key, bucket="platform"):
        return f"https://objects.test/{bucket}/{key}?signature=x"

    def start_multipart_upload(self, bucket, key):
        upload_id = f"multipart-{len(self.multipart_uploads) + 1}"
        self.multipart_uploads[upload_id] = []
        return upload_id

    def part_upload_urls(self, bucket, key, upload_id, parts):
        urls = [
            f"https://objects.test/{bucket}/{key}?uploadId={upload_id}&partNumber={number}"
            for number in range(1, parts + 1)
        ]
        self.multipart_uploads[upload_id] = urls
        return urls

    def receive_part(self, url, content):
        self.parts[url] = content

    def complete_multipart_upload(self, bucket, key, upload_id):
        urls = self.multipart_uploads.pop(upload_id)
        self.buckets[bucket][key] = b"".join(self.parts.pop(url, b"") for url in urls)

    def read(self, bucket, key):
        return self.buckets[bucket][key]

    def objects_under(self, bucket, prefix):
        return {k: v for k, v in self.buckets[bucket].items() if k.startswith(prefix)}

    def list_objects(self, bucket, prefix):
        return [{"Key": k, "Size": len(v)} for k, v in self.objects_under(bucket, prefix).items()]

    def size_of(self, bucket, prefix):
        return sum(len(v) for v in self.objects_under(bucket, prefix).values())

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

    def create_model_version(self, name, artifact_prefix, tags):
        version = 1 + max((v.version for v in self.versions if v.name == name), default=0)
        self.versions.append(ModelVersion(name, version, tags, artifact_prefix))
        return version

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
    """Records Secrets, Kubeflow runs and Endpoints; tests set run, pod and Endpoint states."""

    steps = StepEnvironment(
        stages_image="mlp-stages:test",
        trainer_images={"hf": "mlp-trainer-hf:test", "unsloth": "mlp-trainer-unsloth:test"},
        model_cache_pvc="model-cache",
        object_store_url="http://seaweedfs.test:8333",
        object_store_bucket="platform",
        api_url="http://api.mlp.test:8000",
        sandbox_url="http://sandbox.mlp.test:8090",
        platform_namespace="mlp",
    )
    endpoint_environment = EndpointEnvironment(
        namespace="mlp",
        domain="platform.test",
        vllm_image="vllm/vllm-openai:test",
        stages_image="mlp-stages:test",
        model_cache_host_path="/var/lib/mlp/model-cache",
        object_store_url="http://seaweedfs.test:8333",
    )

    def __init__(self):
        self.secrets = {}
        self.secret_created = {}
        self.runs = {}
        self.terminated = []
        self.waiting_for_gpu = set()
        # Endpoint name -> its Kubernetes objects, and the state its pod is in.
        self.endpoints = {}
        self.endpoint_states = {}
        self.deleted_endpoints = []

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

    def create_endpoint(self, manifests):
        name = manifests["deployment"]["metadata"]["labels"][config.ENDPOINT_LABEL]
        self.endpoints[name] = manifests

    def delete_endpoint(self, name):
        self.endpoints.pop(name, None)
        self.deleted_endpoints.append(name)

    def endpoint_state(self, name):
        return self.endpoint_states.get(name, "pending") if name in self.endpoints else None


@pytest.fixture
def cluster(api):
    api.app.state.cluster = FakeCluster()
    return api.app.state.cluster


@pytest.fixture
def model_cache(api, tmp_path):
    """The Model Cache, empty until a test writes into it."""
    api.app.state.model_cache = tmp_path / "model-cache"
    return api.app.state.model_cache


@pytest.fixture
def logged_in_api(api, hugging_face, object_store, model_registry, cluster, model_cache):
    api.post(api_paths.LOGIN, json={"password": PASSWORD})
    return api
