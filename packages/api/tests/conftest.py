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
    monkeypatch.setenv("GPU_QUEUE", "gpu")
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
        # Dataset repo -> its files.
        self.dataset_files = {}
        # What every search finds, and each search's word and token.
        self.search_results = []
        self.searches = []

    def find_model(self, repo, revision, token):
        self.lookups.append((repo, revision, token))
        return self.models.get(repo)

    def model_size(self, repo, commit, token):
        return self.sizes.get(repo)

    def model_file(self, repo, commit, path, token):
        return self.files.get(repo, {}).get(path)

    def dataset_file(self, repo, commit, path):
        return self.dataset_files[repo][path]

    def search_models(self, word, token):
        self.searches.append((word, token))
        return self.search_results

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

    def write(self, key, content, bucket="platform"):
        self.buckets[bucket][key] = content

    def delete(self, key, bucket="platform"):
        self.buckets[bucket].pop(key, None)

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
    """Model Versions, files in the fake object store's mlflow bucket, and experiments' metrics."""

    artifact_bucket = "mlflow"

    def __init__(self):
        self.versions = []
        self.deleted_models = []
        # MLflow Run ID -> where its artifacts lie in the mlflow bucket.
        self.runs = {}
        # Experiment name -> Run name -> Run ID, and Run ID -> its logged metrics.
        self.experiments = {}
        self.metrics = {}
        self.history_reads = []
        # Experiment ID -> its Request Traces, as `traces/batchGet` answers them; and each read.
        self.traces = {}
        self.trace_searches = []
        self.trace_reads = []
        # Each delete of traces, Runs and experiments, in order.
        self.deletions = []

    def model_versions(self):
        return list(self.versions)

    def run_artifact_prefix(self, run_id):
        return self.runs.get(run_id)

    def create_model_version(self, name, artifact_prefix, tags):
        version = 1 + max((v.version for v in self.versions if v.name == name), default=0)
        self.versions.append(ModelVersion(name, version, tags, artifact_prefix))
        return version

    def delete_model_version(self, name, version):
        self.versions = [v for v in self.versions if (v.name, v.version) != (name, version)]

    def delete_registered_model(self, name):
        self.versions = [v for v in self.versions if v.name != name]
        self.deleted_models.append(name)

    # Experiments are known by their name.
    def experiment_id(self, name, create=True):
        if create:
            self.experiments.setdefault(name, {})
        return name if name in self.experiments else None

    def run_ids(self, experiment_id):
        return dict(self.experiments[experiment_id])

    def create_run(self, experiment_id, name, start_time_ms):
        run_id = f"{experiment_id}:{name}"
        self.experiments[experiment_id][name] = run_id
        self.metrics[run_id] = []
        return run_id

    def log_metrics(self, run_id, values, step, time_ms):
        self.metrics[run_id] += [(key, value, step) for key, value in values.items()]

    # Every point within the steps, without MLflow's downsampling.
    def metric_history(self, run_ids, key, steps, max_results):
        self.history_reads.append((run_ids, key, steps, max_results))
        return sorted(
            (step, value)
            for run_id in run_ids
            for found, value, step in self.metrics[run_id]
            if found == key and steps[0] <= step <= steps[1]
        )

    def search_traces(self, experiment_id, since_ms, order_by, max_results):
        self.trace_searches.append((experiment_id, since_ms, order_by, max_results))
        orders = {
            "timestamp_ms DESC": lambda info: datetime.fromisoformat(info["request_time"]),
            "execution_time_ms DESC": lambda info: float(info["execution_duration"][:-1]),
        }
        since = datetime.fromtimestamp(since_ms / 1000, UTC)
        infos = [trace["trace_info"] for trace in self.traces.get(experiment_id, [])]
        found = [info for info in infos if datetime.fromisoformat(info["request_time"]) >= since]
        return sorted(found, key=orders[order_by], reverse=True)[:max_results]

    def batch_get_traces(self, trace_ids):
        self.trace_reads.append(trace_ids)
        every = [trace for traces in self.traces.values() for trace in traces]
        return [trace for trace in every if trace["trace_info"]["trace_id"] in trace_ids]

    def experiment_ids(self, prefix):
        return [name for name in self.experiments if name.startswith(prefix)]

    def delete_traces(self, experiment_id, max_timestamp_ms):
        self.deletions.append(("traces", experiment_id, max_timestamp_ms))
        before = datetime.fromtimestamp(max_timestamp_ms / 1000, UTC)
        self.traces[experiment_id] = [
            trace
            for trace in self.traces.get(experiment_id, [])
            if datetime.fromisoformat(trace["trace_info"]["request_time"]) > before
        ]

    def delete_run(self, run_id):
        self.deletions.append(("run", run_id))
        for runs in self.experiments.values():
            for name in [name for name, found in runs.items() if found == run_id]:
                del runs[name]

    def rename_experiment(self, experiment_id, new_name):
        self.deletions.append(("rename", experiment_id, new_name))

    # A deleted experiment's ID stays unknown, so a reused name gets a fresh one.
    def delete_experiment(self, experiment_id):
        self.deletions.append(("experiment", experiment_id))
        del self.experiments[experiment_id]

    def logged(self, endpoint_name):
        """Each logged Server Sample of the Endpoint name, by Run name: step -> key -> value."""
        runs = self.experiments.get(f"endpoint/{endpoint_name}", {})
        logged = {}
        for run_name, run_id in runs.items():
            samples = logged.setdefault(run_name, {})
            for key, value, step in self.metrics[run_id]:
                samples.setdefault(step, {})[key] = value
        return logged


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
        gpu_queue="gpu",
    )
    endpoint_environment = EndpointEnvironment(
        namespace="mlp",
        domain="platform.test",
        vllm_image="vllm/vllm-openai:test",
        stages_image="mlp-stages:test",
        model_cache_host_path="/var/lib/mlp/model-cache",
        object_store_url="http://seaweedfs.test:8333",
        mlflow_url="http://mlflow.test",
        gpu_queue="gpu",
    )

    def __init__(self):
        self.secrets = {}
        self.secret_created = {}
        self.runs = {}
        self.terminated = []
        self.deleted_runs = []
        self.waiting_for_gpu = set()
        # Endpoint name -> its Kubernetes objects, and the state its pod is in.
        self.endpoints = {}
        self.endpoint_states = {}
        self.deleted_endpoints = []
        # Endpoint name -> its vLLM's `/metrics` text; any other Endpoint is unreachable.
        self.metrics = {}
        self.chat_requests = []

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

    def delete_run(self, run_id):
        self.runs.pop(run_id, None)
        self.deleted_runs.append(run_id)

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

    def endpoint_metrics(self, name):
        if name not in self.metrics:
            raise ConnectionError(f"endpoint-{name} timed out")
        return self.metrics[name]

    def send_chat_request(self, name, request, key):
        if name not in self.metrics:
            raise ConnectionError(f"endpoint-{name} timed out")
        self.chat_requests.append((name, request, key))


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
