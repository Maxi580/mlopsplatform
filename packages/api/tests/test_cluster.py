import json
from types import SimpleNamespace

from kubernetes import client

from mlp_api.pipelines.cluster import Cluster

from .conftest import FakeCluster


def kubernetes_requests(cluster_call) -> list[tuple[str, str]]:
    """The method and path of each request the real Kubernetes client sends for the call."""
    requests = []

    def request(method, url, **_):
        requests.append((method, url.removeprefix("http://kubernetes.test")))
        return SimpleNamespace(status=200, data=b"{}", getheaders=lambda: {})

    cluster = Cluster.__new__(Cluster)
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


def pending_pods(*conditions: dict) -> Cluster:
    """A cluster whose run has one Pending pod with the given conditions."""
    pods = {"items": [{"status": {"phase": "Pending", "conditions": list(conditions)}}]}

    def request(method, url, **_):
        return SimpleNamespace(status=200, data=json.dumps(pods).encode(), getheaders=lambda: {})

    cluster = Cluster.__new__(Cluster)
    cluster.namespace = "kubeflow"
    cluster.kubernetes_client = client.ApiClient(
        client.Configuration(host="http://kubernetes.test")
    )
    cluster.kubernetes_client.request = request
    return cluster


def test_a_step_kueue_holds_back_until_a_gpu_is_free_waits_for_a_gpu():
    gated = {"type": "PodScheduled", "status": "False", "reason": "SchedulingGated"}

    assert pending_pods(gated).is_waiting_for_gpu("run-1")


def test_a_step_the_scheduler_finds_no_gpu_for_waits_for_a_gpu():
    message = "0/1 nodes are available: 1 Insufficient nvidia.com/gpu."
    unschedulable = {"type": "PodScheduled", "status": "False", "reason": "Unschedulable"}

    assert pending_pods({**unschedulable, "message": message}).is_waiting_for_gpu("run-1")
    assert not pending_pods(unschedulable).is_waiting_for_gpu("run-1")
