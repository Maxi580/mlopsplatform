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
