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
# What every Endpoint answers a chat request with in tests.
CHAT_ANSWER = {
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi!"}}],
}
JWT_SECRET = "test-jwt-secret-of-at-least-32-bytes"


@pytest.fixture
def platform_database(settings_configmap_env, tmp_path, monkeypatch):
    database = tmp_path / "platform.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database}")
    monkeypatch.setenv("JWT_SECRET", JWT_SECRET)
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://seaweedfs.test:8333")
    monkeypatch.setenv("S3_BUCKET", "platform")
    monkeypatch.setenv("MLFLOW_URL", "http://mlflow.test")
    monkeypatch.setenv("MLFLOW_BUCKET", "mlflow")
    monkeypatch.setenv("PROXY_URL", "http://proxy.mlp.test:8010")
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
    monkeypatch.setenv("LAUNCHER_IMAGE", "mlp-launcher:real")
    monkeypatch.setenv("FINETUNE_RUNTIME", "mlp-finetune")
    # Tests reconcile by hand, so the background loop never races them.
    monkeypatch.setattr(config, "RECONCILE_INTERVAL", timedelta(days=1))
    monkeypatch.setattr(config, "EVICTION_INTERVAL", timedelta(days=1))
    monkeypatch.setattr(config, "JUDGING_INTERVAL", timedelta(days=1))
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

    def download_url(self, key, address, bucket="platform"):
        return f"https://{address}/{bucket}/{key}?signature=x"

    def start_multipart_upload(self, bucket, key):
        upload_id = f"multipart-{len(self.multipart_uploads) + 1}"
        self.multipart_uploads[upload_id] = []
        return upload_id

    def part_upload_urls(self, bucket, key, upload_id, parts, address):
        urls = [
            f"https://{address}/{bucket}/{key}?uploadId={upload_id}&partNumber={number}"
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

    def every_metric_point(self, run_id, key):
        return sorted((step, value) for found, value, step in self.metrics[run_id] if found == key)

    # Pages by offset, as MLflow's page tokens are opaque.
    def search_traces(self, experiment_id, clauses, order_by, max_results, page_token=None):
        self.trace_searches.append((experiment_id, clauses, order_by, max_results, page_token))
        infos = [trace["trace_info"] for trace in self.traces.get(experiment_id, [])]
        found = [info for info in infos if all(matches(info, *clause) for clause in clauses)]
        key, direction = order_by.split()
        found.sort(key=lambda info: search_value(info, f"trace.{key}"), reverse=direction == "DESC")
        start = int(page_token or 0)
        more = start + max_results < len(found)
        return found[start : start + max_results], str(start + max_results) if more else None

    def batch_get_traces(self, trace_ids):
        self.trace_reads.append(trace_ids)
        every = [trace for traces in self.traces.values() for trace in traces]
        return [trace for trace in every if trace["trace_info"]["trace_id"] in trace_ids]

    def trace_info(self, trace_id):
        every = [trace["trace_info"] for traces in self.traces.values() for trace in traces]
        return next((info for info in every if info["trace_id"] == trace_id), None)

    def set_trace_tag(self, trace_id, key, value):
        self.trace_info(trace_id)["tags"][key] = value

    # One assessment per name, as MLflow's update replaces it.
    def set_feedback(self, info, name, value, source_type, rationale=None):
        assessments = self.trace_info(info["trace_id"])["assessments"]
        assessments[:] = [a for a in assessments if a["assessment_name"] != name]
        assessment = {"assessment_name": name, "feedback": {"value": value}}
        source = {"source": {"source_type": source_type}}
        assessments.append(
            {**assessment, **source, **({"rationale": rationale} if rationale else {})}
        )

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


def search_value(info: dict, key: str):
    """What MLflow's trace search compares for the key: times in ms, tags, assessments."""
    if key == "trace.timestamp_ms":
        return datetime.fromisoformat(info["request_time"]).timestamp() * 1000
    if key == "trace.execution_time_ms":
        return float(info["execution_duration"].removesuffix("s")) * 1000
    if key.startswith("tag."):
        return info.get("tags", {}).get(key.removeprefix("tag."))
    name = key.removeprefix("feedback.")
    found = [a for a in info.get("assessments", []) if a["assessment_name"] == name]
    return found[0]["feedback"]["value"] if found else None


def matches(info: dict, key: str, comparator: str, value) -> bool:
    found = search_value(info, key)
    if comparator in ("IS NULL", "IS NOT NULL"):
        return (found is None) == (comparator == "IS NULL")
    if found is None:
        return False
    return {"=": found == value, ">=": found >= value, "<=": found <= value}[comparator]


def otel_value(value) -> dict:
    """The value as MLflow 3.16's `traces/batchGet` writes an OTel AnyValue to JSON."""
    if isinstance(value, dict):
        pairs = [{"key": key, "value": otel_value(found)} for key, found in value.items()]
        return {"kvlist_value": {"values": pairs}}
    if isinstance(value, list):
        return {"array_value": {"values": [otel_value(found) for found in value]}}
    if isinstance(value, bool):
        return {"bool_value": value}
    # An int64 MLflow writes as a JSON string, which tests give as a string of digits.
    if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
        return {"int_value": value}
    if isinstance(value, float):
        return {"double_value": value}
    return {} if value is None else {"string_value": value}


def span(name: str, attributes: dict) -> dict:
    return {
        "name": name,
        "attributes": [{"key": key, "value": otel_value(v)} for key, v in attributes.items()],
    }


def record_request_trace(
    model_registry, experiment_id, trace_id, arrived, e2e, request, response, tags=None, **sampling
):
    """A Request Trace: the Endpoint Proxy's root span and vLLM's span inside it, as MLflow 3.16
    answers `traces/batchGet`; int64 values come as JSON strings."""
    vllm = {
        "gen_ai.latency.e2e": e2e,
        "gen_ai.latency.time_to_first_token": 0.25,
        "gen_ai.latency.time_in_queue": 0.125,
        "gen_ai.latency.time_in_model_prefill": 0.0625,
        "gen_ai.latency.time_in_model_decode": e2e - 0.25,
        "gen_ai.latency.time_in_model_inference": e2e - 0.125,
        "gen_ai.usage.prompt_tokens": "12",
        "gen_ai.usage.completion_tokens": 40,
        "gen_ai.request.id": f"chatcmpl-{trace_id}",
        **sampling,
    }
    root = {"mlflow.spanInputs": request, "mlflow.spanOutputs": response, "http.status_code": 200}
    location = {"type": "MLFLOW_EXPERIMENT", "mlflow_experiment": {"experiment_id": experiment_id}}
    model_registry.traces.setdefault(experiment_id, []).append(
        {
            "trace_info": {
                "trace_id": trace_id,
                "trace_location": location,
                "request_time": arrived.isoformat().replace("+00:00", "Z"),
                "execution_duration": f"{e2e}s",
                "state": "OK",
                "tags": {"source": "api", "status": "ok", **(tags or {})},
                "assessments": [],
            },
            "spans": [span("endpoint_request", root), span("llm_request", vllm)],
        }
    )


class FakeCluster:
    """Records Secrets, Kubeflow runs and Endpoints; tests set run, pod and Endpoint states."""

    steps = StepEnvironment(
        stages_image="mlp-stages:test",
        launcher_image="mlp-launcher:test",
        finetune_runtime="mlp-finetune",
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
        proxy_url="http://proxy.mlp.test:8010",
        gpu_queue="gpu",
    )

    def __init__(self):
        self.secrets = {}
        self.secret_created = {}
        self.runs = {}
        self.terminated = []
        self.deleted_runs = []
        self.waiting_for_gpu = set()
        # Pipeline ids that have TrainJobs, and those of them Kueue holds back.
        self.trainjobs = set()
        self.queued_trainjobs = set()
        # Run ID -> the log of its first failed step.
        self.failed_step_logs = {}
        # Endpoint name -> its Kubernetes objects, and the state its pod is in.
        self.endpoints = {}
        self.endpoint_states = {}
        self.deleted_endpoints = []
        # Endpoint name -> what Kubernetes still has of its logs, and those scaled down.
        self.logs = {}
        self.scaled_to_zero = []
        # Endpoint name -> its vLLM's `/metrics` text; any other Endpoint is unreachable.
        self.metrics = {}
        self.chat_requests = []
        # Each request sent to an Endpoint's Service directly, and what an Endpoint answers it.
        self.asked = []
        self.answers = {}

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

    def is_waiting_for_gpu(self, run_id, pipeline_id):
        return run_id in self.waiting_for_gpu or pipeline_id in self.queued_trainjobs

    def failed_step_log(self, run_id):
        return self.failed_step_logs.get(run_id, "")

    def trainjob_pipelines(self):
        return set(self.trainjobs)

    def delete_trainjobs(self, pipeline_id):
        self.trainjobs.discard(pipeline_id)

    def create_endpoint(self, manifests):
        name = manifests["deployment"]["metadata"]["labels"][config.ENDPOINT_LABEL]
        self.endpoints[name] = manifests

    def delete_endpoint(self, name):
        self.endpoints.pop(name, None)
        self.deleted_endpoints.append(name)

    def endpoint_state(self, name):
        return self.endpoint_states.get(name, "pending") if name in self.endpoints else None

    def endpoint_logs(self, name):
        return self.logs.get(name, "")

    def scale_endpoint_to_zero(self, name):
        self.scaled_to_zero.append(name)
        self.endpoint_states[name] = "failed"

    def endpoint_metrics(self, name):
        if name not in self.metrics:
            raise ConnectionError(f"endpoint-{name} timed out")
        return self.metrics[name]

    # Reachable Endpoints answer from `answers`, else with CHAT_ANSWER.
    def ask_endpoint(self, name, request, timeout):
        if name not in self.metrics:
            raise ConnectionError(f"endpoint-{name} timed out")
        self.asked.append((name, request))
        return self.answers.get(name, CHAT_ANSWER)

    # Through the Endpoint Proxy, which with vLLM records the request's trace in full.
    def send_chat_request(self, name, request, experiment_id):
        if name not in self.metrics:
            raise ConnectionError(f"endpoint-{name} timed out")
        self.chat_requests.append((name, request, experiment_id))
        trace_id = f"tr-{name}-{len(self.chat_requests)}"
        sent = {"model": name, **request}
        record_request_trace(
            self.model_registry, experiment_id, trace_id, datetime.now(UTC), 0.5, sent, CHAT_ANSWER
        )
        return trace_id


@pytest.fixture
def cluster(api, model_registry):
    api.app.state.cluster = FakeCluster()
    api.app.state.cluster.model_registry = model_registry
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
