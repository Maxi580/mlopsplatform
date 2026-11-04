import json
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest
from kubernetes import client
from kubernetes.client.exceptions import ApiException

from mlp_api.pipelines.cluster import Cluster
from mlp_core import config

from .conftest import FakeCluster


def kubernetes_requests(cluster_call) -> list[tuple[str, str]]:
    """The method and path, with its query, of each request the real Kubernetes client sends."""
    requests = []

    def request(method, url, query_params=None, **_):
        query = f"?{urlencode(query_params)}" if query_params else ""
        requests.append((method, url.removeprefix("http://kubernetes.test") + query))
        return SimpleNamespace(status=200, data=b"{}", getheaders=lambda: {})

    cluster = Cluster.__new__(Cluster)
    cluster.namespace = "kubeflow"
    cluster.endpoint_environment = FakeCluster.endpoint_environment
    cluster.kubernetes_client = client.ApiClient(
        client.Configuration(host="http://kubernetes.test")
    )
    cluster.kubernetes_client.request = request
    cluster_call(cluster)
    return requests


def test_an_endpoint_starts_with_its_deployment_service_and_route():
    manifests = {"deployment": {}, "service": {}, "route": {}}

    requests = kubernetes_requests(lambda cluster: cluster.create_endpoint(manifests))

    assert requests == [
        ("POST", "/apis/apps/v1/namespaces/mlp/deployments"),
        ("POST", "/api/v1/namespaces/mlp/services"),
        ("POST", "/apis/traefik.io/v1alpha1/namespaces/mlp/ingressroutes"),
    ]


def test_an_endpoint_stops_by_deleting_its_deployment_service_and_route():
    requests = kubernetes_requests(lambda cluster: cluster.delete_endpoint("chat"))

    assert requests == [
        ("DELETE", "/apis/apps/v1/namespaces/mlp/deployments/endpoint-chat"),
        ("DELETE", "/api/v1/namespaces/mlp/services/endpoint-chat"),
        ("DELETE", "/apis/traefik.io/v1alpha1/namespaces/mlp/ingressroutes/endpoint-chat"),
    ]


def cluster_listing(conditions: list[dict] = (), trainjobs: list[dict] = ()) -> Cluster:
    """A cluster whose run has one Pending pod with the conditions, and the Pipeline TrainJobs."""
    pods = {"items": [{"status": {"phase": "Pending", "conditions": list(conditions)}}]}

    def request(method, url, **_):
        found = {"items": list(trainjobs)} if "/trainjobs" in url else pods
        return SimpleNamespace(status=200, data=json.dumps(found).encode(), getheaders=lambda: {})

    cluster = Cluster.__new__(Cluster)
    cluster.namespace = "kubeflow"
    cluster.kubernetes_client = client.ApiClient(
        client.Configuration(host="http://kubernetes.test")
    )
    cluster.kubernetes_client.request = request
    return cluster


def test_a_step_kueue_holds_back_until_a_gpu_is_free_waits_for_a_gpu():
    gated = {"type": "PodScheduled", "status": "False", "reason": "SchedulingGated"}

    assert cluster_listing([gated]).is_waiting_for_gpu("run-1", 7)


def test_a_step_the_scheduler_finds_no_gpu_for_waits_for_a_gpu():
    message = "0/1 nodes are available: 1 Insufficient nvidia.com/gpu."
    unschedulable = {"type": "PodScheduled", "status": "False", "reason": "Unschedulable"}

    assert cluster_listing([{**unschedulable, "message": message}]).is_waiting_for_gpu("run-1", 7)
    assert not cluster_listing([unschedulable]).is_waiting_for_gpu("run-1", 7)


def test_a_phase_whose_trainjob_kueue_holds_back_waits_for_a_gpu():
    queued = {"metadata": {"name": "pipeline-7-phase-0-x1"}, "spec": {"suspend": True}}
    admitted = {"metadata": {"name": "pipeline-7-phase-0-x1"}, "spec": {"suspend": False}}

    assert cluster_listing(trainjobs=[queued]).is_waiting_for_gpu("run-1", 7)
    assert not cluster_listing(trainjobs=[admitted]).is_waiting_for_gpu("run-1", 7)


def test_the_trainjobs_of_a_pipeline_are_found_and_deleted_by_its_label():
    requests = kubernetes_requests(lambda cluster: cluster.delete_trainjobs(7))

    assert requests == [
        (
            "DELETE",
            "/apis/trainer.kubeflow.org/v1alpha1/namespaces/kubeflow/trainjobs"
            "?labelSelector=mlp%2Fpipeline-id%3D7",
        )
    ]


UNSCHEDULABLE = {
    "type": "PodScheduled",
    "status": "False",
    "reason": "Unschedulable",
    "message": "0/1 nodes are available: 1 Insufficient nvidia.com/gpu.",
}
READY = {"type": "Ready", "status": "True"}
BACKING_OFF = {"waiting": {"reason": "CrashLoopBackOff"}}


def container(name="vllm", state=None, restarts=0) -> dict:
    return {
        "name": name,
        "image": "vllm",
        "imageID": "vllm@sha",
        "ready": False,
        "restartCount": restarts,
        "state": state or {"running": {}},
    }


def log_answer(log: bytes | None) -> SimpleNamespace:
    """The log, or the 404 the real client raises once Kubernetes no longer has it."""
    if log is None:
        raise ApiException(status=404)
    return SimpleNamespace(status=200, data=log, getheaders=lambda: {})


def endpoint_cluster(
    replicas=1, conditions=(), containers=(), init_containers=(), logs=None
) -> Cluster:
    """The Deployment at the replicas, one pod as given, and logs by (container, previous)."""
    deployment = {"spec": {"replicas": replicas, "selector": {}, "template": {}}}
    pod = {
        "metadata": {"name": "endpoint-chat-1"},
        "status": {
            "conditions": list(conditions),
            "containerStatuses": list(containers),
            "initContainerStatuses": list(init_containers),
        },
    }

    def request(method, url, query_params=None, **_):
        if url.endswith("/log"):
            query = dict(query_params)
            return log_answer((logs or {}).get((query["container"], query["previous"])))
        found = deployment if "/deployments/" in url else {"items": [pod]}
        return SimpleNamespace(status=200, data=json.dumps(found).encode(), getheaders=lambda: {})

    cluster = Cluster.__new__(Cluster)
    cluster.endpoint_environment = FakeCluster.endpoint_environment
    cluster.kubernetes_client = client.ApiClient(
        client.Configuration(host="http://kubernetes.test")
    )
    cluster.kubernetes_client.request = request
    return cluster


@pytest.mark.parametrize(
    "condition",
    [{"type": "PodScheduled", "status": "False", "reason": "SchedulingGated"}, UNSCHEDULABLE],
    ids=["gated", "unschedulable"],
)
def test_an_endpoint_held_back_for_gpus_is_waiting_for_a_gpu(condition):
    assert endpoint_cluster(conditions=[condition]).endpoint_state("chat") == "waiting for GPU"


def test_an_endpoint_unschedulable_for_another_reason_is_pending():
    other = {**UNSCHEDULABLE, "message": "0/1 nodes are available: 1 node is cordoned."}

    assert endpoint_cluster(conditions=[other]).endpoint_state("chat") == "pending"


def test_an_endpoint_at_zero_replicas_stays_failed():
    assert endpoint_cluster(replicas=0).endpoint_state("chat") == "failed"


def test_an_endpoint_with_its_gpu_share_that_is_loading_is_pending():
    assert endpoint_cluster(containers=[container()]).endpoint_state("chat") == "pending"


def test_an_endpoint_restarted_by_a_reboot_does_not_count_as_crashing():
    assert endpoint_cluster(containers=[container(restarts=1)]).endpoint_state("chat") == "pending"


def test_a_ready_endpoint_that_restarted_is_running():
    cluster = endpoint_cluster(conditions=[READY], containers=[container(restarts=3)])

    assert cluster.endpoint_state("chat") == "running"


@pytest.mark.parametrize("failing", ["containers", "init_containers"])
def test_an_endpoint_kubernetes_backs_off_restarting_is_crashing(failing):
    cluster = endpoint_cluster(**{failing: [container(state=BACKING_OFF, restarts=3)]})

    assert cluster.endpoint_state("chat") == config.ENDPOINT_CRASHING


def test_scaling_an_endpoint_to_zero_keeps_its_deployment():
    requests = kubernetes_requests(lambda cluster: cluster.scale_endpoint_to_zero("chat"))

    assert requests == [("PATCH", "/apis/apps/v1/namespaces/mlp/deployments/endpoint-chat/scale")]


def test_a_crashed_endpoints_log_has_every_container_oldest_attempt_first():
    cluster = endpoint_cluster(
        init_containers=[container("fetch")],
        containers=[container(state=BACKING_OFF, restarts=2)],
        logs={
            ("fetch", False): b"fetched",
            ("vllm", True): b"attempt 1",
            ("vllm", False): b"attempt 2",
        },
    )

    log = cluster.endpoint_logs("chat")

    assert log.index("fetched") < log.index("attempt 1") < log.index("attempt 2")


def test_an_attempt_kubernetes_answers_twice_while_backing_off_is_logged_once():
    cluster = endpoint_cluster(
        containers=[container(state=BACKING_OFF, restarts=2)],
        logs={("vllm", True): b"the crash", ("vllm", False): b"the crash"},
    )

    assert cluster.endpoint_logs("chat").count("the crash") == 1


def test_logs_kubernetes_no_longer_has_are_left_out():
    cluster = endpoint_cluster(containers=[container(state=BACKING_OFF, restarts=2)])

    assert cluster.endpoint_logs("chat") == ""


def test_a_log_that_is_not_utf8_is_still_read():
    cluster = endpoint_cluster(
        containers=[container(state=BACKING_OFF, restarts=2)],
        logs={("vllm", False): b"caf\xe9 crashed"},
    )

    assert "crashed" in cluster.endpoint_logs("chat")


def run_with_failed_pods(pods: list[dict], logs: dict[str, bytes]) -> Cluster:
    """A cluster whose run has the Failed pods, with `main` logs by pod name."""

    def request(method, url, **_):
        if url.endswith("/log"):
            return log_answer(logs.get(url.split("/")[-2]))
        answer = json.dumps({"items": pods}).encode()
        return SimpleNamespace(status=200, data=answer, getheaders=lambda: {})

    cluster = Cluster.__new__(Cluster)
    cluster.namespace = "kubeflow"
    cluster.kubernetes_client = client.ApiClient(
        client.Configuration(host="http://kubernetes.test")
    )
    cluster.kubernetes_client.request = request
    return cluster


def failed_pod(name: str, created: str) -> dict:
    return {"metadata": {"name": name, "creationTimestamp": created}}


def test_a_failed_run_is_explained_by_the_log_of_its_first_failed_step():
    pods = [
        failed_pod("later", "2026-10-02T12:05:00Z"),
        failed_pod("first", "2026-10-02T12:00:00Z"),
    ]
    logs = {"first": b"the cause", "later": b"a consequence"}

    assert run_with_failed_pods(pods, logs).failed_step_log("run-1") == "the cause"


@pytest.mark.parametrize(
    "pods",
    [[], [failed_pod("cleaned-up", "2026-10-02T12:00:00Z")]],
    ids=["no pod", "log gone"],
)
def test_a_failed_run_whose_pods_or_logs_are_gone_has_no_log(pods):
    assert run_with_failed_pods(pods, {}).failed_step_log("run-1") == ""
