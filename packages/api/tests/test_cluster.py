import json
from types import SimpleNamespace
from urllib.parse import urlencode

from kubernetes import client

from mlp_api.pipelines.cluster import Cluster

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
