import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import cached_property

import httpx2 as httpx
from kubernetes import client
from kubernetes import config as kubernetes_config
from kubernetes.client.exceptions import ApiException

from mlp_api.pipelines.compiler import StepEnvironment
from mlp_core import config


@dataclass(frozen=True)
class KubeflowRun:
    state: str
    mlflow_run_url: str | None


class Cluster:
    """Kubeflow runs and the Kubernetes objects of their steps; tests swap in a fake."""

    def __init__(self):
        self.namespace = os.environ["KUBEFLOW_NAMESPACE"]
        self.steps = StepEnvironment.from_environment()
        self.kubeflow = httpx.Client(base_url=f"{os.environ['KUBEFLOW_URL']}/apis/v2beta1")

    # Loaded on first use, so the API also starts outside a cluster.
    @cached_property
    def kubernetes(self) -> client.CoreV1Api:
        kubernetes_config.load_incluster_config()
        return client.CoreV1Api()

    def create_secret(self, name: str, values: dict[str, str]) -> None:
        secret = client.V1Secret(
            metadata=client.V1ObjectMeta(name=name, labels={config.PIPELINE_SECRET_LABEL: "true"}),
            string_data=values,
        )
        self.kubernetes.create_namespaced_secret(self.namespace, secret)

    def delete_secret(self, name: str) -> None:
        try:
            self.kubernetes.delete_namespaced_secret(name, self.namespace)
        except ApiException as error:
            if error.status != 404:
                raise

    def secret_ages(self) -> dict[str, timedelta]:
        secrets = self.kubernetes.list_namespaced_secret(
            self.namespace, label_selector=config.PIPELINE_SECRET_LABEL
        )
        now = datetime.now(UTC)
        return {s.metadata.name: now - s.metadata.creation_timestamp for s in secrets.items}

    def submit_run(self, display_name: str, pipeline_spec: dict) -> str:
        run = {"display_name": display_name, "pipeline_spec": pipeline_spec}
        response = self.kubeflow.post("/runs", json=run)
        response.raise_for_status()
        return response.json()["run_id"]

    def find_run(self, run_id: str) -> KubeflowRun | None:
        """The run, or None once someone deleted it in the KFP UI."""
        response = self.kubeflow.get(f"/runs/{run_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        run = response.json()
        # KFP's MLflow plugin reports the run's MLflow Run here once it is configured.
        mlflow = run.get("plugins_output", {}).get("mlflow", {}).get("entries", {})
        return KubeflowRun(run["state"], mlflow.get("run_url", {}).get("value"))

    def terminate_run(self, run_id: str) -> None:
        response = self.kubeflow.post(f"/runs/{run_id}:terminate")
        if response.status_code != 404:
            response.raise_for_status()

    def is_waiting_for_gpu(self, run_id: str) -> bool:
        pods = self.kubernetes.list_namespaced_pod(
            self.namespace,
            label_selector=f"pipeline/runid={run_id}",
            field_selector="status.phase=Pending",
        )
        return any(
            condition.reason == "Unschedulable" and config.GPU_RESOURCE in (condition.message or "")
            for pod in pods.items
            for condition in pod.status.conditions or []
        )
