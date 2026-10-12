import os
from dataclasses import dataclass

import httpx2 as httpx

from mlp_core import config


@dataclass(frozen=True)
class ModelVersion:
    name: str
    version: int
    tags: dict[str, str]
    # Where its files lie in the artifact bucket, ending in a slash.
    artifact_prefix: str


class MLflow:
    """MLflow's Model Registry, Runs' artifacts, and experiments' metrics; tests swap in a fake."""

    def __init__(self):
        self.artifact_bucket = os.environ["MLFLOW_BUCKET"]
        self.client = httpx.Client(base_url=f"{os.environ['MLFLOW_URL']}/api/2.0/mlflow")
        # Traces are only in MLflow's 3.0 API.
        self.traces_url = f"{os.environ['MLFLOW_URL']}/api/3.0/mlflow/traces"

    def model_versions(self) -> list[ModelVersion]:
        versions, params = [], {"max_results": config.MLFLOW_PAGE_SIZE}
        while True:
            response = self.client.get("/model-versions/search", params=params)
            response.raise_for_status()
            page = response.json()
            versions += [model_version(found) for found in page.get("model_versions", [])]
            if not page.get("next_page_token"):
                return versions
            params["page_token"] = page["next_page_token"]

    def run_artifact_prefix(self, run_id: str) -> str | None:
        """Where the Run's artifacts lie in the artifact bucket, or None if there is no such Run."""
        response = self.client.get("/runs/get", params={"run_id": run_id})
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return artifact_prefix(response.json()["run"]["info"]["artifact_uri"])

    def create_model_version(self, name: str, artifact_prefix: str, tags: dict[str, str]) -> int:
        """The new version's number, for files already in the artifact bucket."""
        response = self.client.post("/registered-models/create", json={"name": name})
        if response.json().get("error_code") != "RESOURCE_ALREADY_EXISTS":
            response.raise_for_status()
        response = self.client.post(
            "/model-versions/create",
            json={
                "name": name,
                "source": f"mlflow-artifacts:/{artifact_prefix.rstrip('/')}",
                "tags": [{"key": key, "value": value} for key, value in tags.items()],
            },
        )
        response.raise_for_status()
        return int(response.json()["model_version"]["version"])

    def delete_model_version(self, name: str, version: int) -> None:
        response = self.client.request(
            "DELETE", "/model-versions/delete", json={"name": name, "version": str(version)}
        )
        response.raise_for_status()

    def delete_registered_model(self, name: str) -> None:
        response = self.client.request("DELETE", "/registered-models/delete", json={"name": name})
        response.raise_for_status()

    def experiment_id(self, name: str, create: bool = True) -> str | None:
        """The experiment's ID, created if missing; with `create` off, None if missing."""
        response = self.client.get("/experiments/get-by-name", params={"experiment_name": name})
        if response.status_code != 404:
            response.raise_for_status()
            return response.json()["experiment"]["experiment_id"]
        if not create:
            return None
        response = self.client.post("/experiments/create", json={"name": name})
        # Created meanwhile by another caller.
        if response.json().get("error_code") == "RESOURCE_ALREADY_EXISTS":
            return self.experiment_id(name, create=False)
        response.raise_for_status()
        return response.json()["experiment_id"]

    def run_ids(self, experiment_id: str) -> dict[str, str]:
        """The experiment's Runs: name -> ID."""
        response = self.client.post(
            "/runs/search",
            json={"experiment_ids": [experiment_id], "max_results": config.MLFLOW_PAGE_SIZE},
        )
        response.raise_for_status()
        runs = response.json().get("runs", [])
        return {run["info"]["run_name"]: run["info"]["run_id"] for run in runs}

    def create_run(self, experiment_id: str, name: str, start_time_ms: int) -> str:
        response = self.client.post(
            "/runs/create",
            json={"experiment_id": experiment_id, "run_name": name, "start_time": start_time_ms},
        )
        response.raise_for_status()
        return response.json()["run"]["info"]["run_id"]

    def log_metrics(self, run_id: str, values: dict[str, float], step: int, time_ms: int) -> None:
        metrics = [
            {"key": key, "value": value, "step": step, "timestamp": time_ms}
            for key, value in values.items()
        ]
        response = self.client.post("/runs/log-batch", json={"run_id": run_id, "metrics": metrics})
        response.raise_for_status()

    def metric_history(
        self, run_ids: list[str], key: str, steps: tuple[int, int], max_results: int
    ) -> list[tuple[int, float]]:
        """The Runs' (step, value) points within the steps, by step; MLflow downsamples each Run."""
        params = {
            "run_ids": run_ids,
            "metric_key": key,
            "start_step": steps[0],
            "end_step": steps[1],
            "max_results": max_results,
        }
        response = self.client.get("/metrics/get-history-bulk-interval", params=params)
        response.raise_for_status()
        return sorted((m["step"], m["value"]) for m in response.json().get("metrics", []))

    def search_traces(
        self, experiment_id: str, since_ms: int, order_by: str, max_results: int
    ) -> list[dict]:
        """The infos of the experiment's traces since then, in the order; without their spans."""
        location = {"experiment_id": experiment_id}
        body = {
            "locations": [{"type": "MLFLOW_EXPERIMENT", "mlflow_experiment": location}],
            "filter": f"trace.timestamp_ms >= {since_ms}",
            "order_by": [order_by],
            "max_results": max_results,
        }
        response = self.client.post(f"{self.traces_url}/search", json=body)
        response.raise_for_status()
        return response.json().get("traces", [])

    def batch_get_traces(self, trace_ids: list[str]) -> list[dict]:
        """The traces, each its info and spans."""
        response = self.client.get(f"{self.traces_url}/batchGet", params={"trace_ids": trace_ids})
        response.raise_for_status()
        return response.json().get("traces", [])


def model_version(found: dict) -> ModelVersion:
    return ModelVersion(
        name=found["name"],
        version=int(found["version"]),
        tags={tag["key"]: tag["value"] for tag in found.get("tags", [])},
        artifact_prefix=artifact_prefix(found["source"]),
    )


# MLflow proxies artifacts: `mlflow-artifacts:/<path>` lives at `<path>` in its bucket.
def artifact_prefix(uri: str) -> str:
    return uri.removeprefix("mlflow-artifacts:/").rstrip("/") + "/"
