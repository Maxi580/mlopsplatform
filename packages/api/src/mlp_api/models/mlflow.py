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

    def experiment_ids(self, prefix: str) -> list[str]:
        """The IDs of the experiments whose name starts with the prefix, deleted ones left out."""
        experiment_ids = []
        body = {"filter": f"name LIKE '{prefix}%'", "max_results": config.MLFLOW_PAGE_SIZE}
        while True:
            response = self.client.post("/experiments/search", json=body)
            response.raise_for_status()
            page = response.json()
            experiment_ids += [e["experiment_id"] for e in page.get("experiments", [])]
            if not page.get("next_page_token"):
                return experiment_ids
            body["page_token"] = page["next_page_token"]

    # Renamed, as a deleted experiment keeps its name until `mlflow gc`.
    def rename_experiment(self, experiment_id: str, new_name: str) -> None:
        response = self.client.post(
            "/experiments/update", json={"experiment_id": experiment_id, "new_name": new_name}
        )
        response.raise_for_status()

    def delete_experiment(self, experiment_id: str) -> None:
        response = self.client.post("/experiments/delete", json={"experiment_id": experiment_id})
        response.raise_for_status()

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

    def delete_run(self, run_id: str) -> None:
        response = self.client.post("/runs/delete", json={"run_id": run_id})
        response.raise_for_status()

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

    def every_metric_point(self, run_id: str, key: str) -> list[tuple[int, float]]:
        """Every (step, value) point of the Run's metric, by step; read in pages, never sampled."""
        points, params = [], {"run_id": run_id, "metric_key": key, "max_results": 25000}
        while True:
            response = self.client.get("/metrics/get-history", params=params)
            response.raise_for_status()
            page = response.json()
            points += [(m["step"], m["value"]) for m in page.get("metrics", [])]
            if not page.get("next_page_token"):
                return sorted(points)
            params["page_token"] = page["next_page_token"]

    def search_traces(
        self,
        experiment_id: str,
        clauses: list[tuple[str, str, int | float | str | None]],
        order_by: str,
        max_results: int,
        page_token: str | None = None,
    ) -> tuple[list[dict], str | None]:
        """One page of the experiment's matching trace infos, and the next page's token."""
        # A clause is (key, comparator, value), e.g. ("tag.rating", "IS NULL", None).
        location = {"experiment_id": experiment_id}
        body = {
            "locations": [{"type": "MLFLOW_EXPERIMENT", "mlflow_experiment": location}],
            "filter": " AND ".join(filter_clause(*clause) for clause in clauses),
            "order_by": [order_by],
            "max_results": max_results,
            **({"page_token": page_token} if page_token else {}),
        }
        response = self.client.post(f"{self.traces_url}/search", json=body)
        response.raise_for_status()
        page = response.json()
        return page.get("traces", []), page.get("next_page_token") or None

    def batch_get_traces(self, trace_ids: list[str]) -> list[dict]:
        """The traces, each its info and spans."""
        response = self.client.get(f"{self.traces_url}/batchGet", params={"trace_ids": trace_ids})
        response.raise_for_status()
        return response.json().get("traces", [])

    def trace_info(self, trace_id: str) -> dict | None:
        """The trace's info, with its tags and assessments; None if there is no such trace."""
        response = self.client.get(f"{self.traces_url}/{trace_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()["trace"]["trace_info"]

    def set_trace_tag(self, trace_id: str, key: str, value: str) -> None:
        response = self.client.patch(f"/traces/{trace_id}/tags", json={"key": key, "value": value})
        response.raise_for_status()

    def set_feedback(
        self, info: dict, name: str, value, source_type: str, rationale: str | None = None
    ) -> None:
        """Sets the trace's feedback assessment of the name, replacing the one it had."""
        assessment = {
            "trace_id": info["trace_id"],
            "assessment_name": name,
            "source": {"source_type": source_type, "source_id": source_type.lower()},
            "feedback": {"value": value},
            **({"rationale": rationale} if rationale else {}),
        }
        url = f"{self.traces_url}/{info['trace_id']}/assessments"
        found = [a for a in info.get("assessments", []) if a["assessment_name"] == name]
        if found:
            assessment["assessment_id"] = found[0]["assessment_id"]
            body = {"assessment": assessment, "update_mask": "feedback,rationale"}
            response = self.client.patch(f"{url}/{found[0]['assessment_id']}", json=body)
        else:
            response = self.client.post(url, json={"assessment": assessment})
        response.raise_for_status()

    # Unlike Runs and experiments, traces are deleted for good, with their spans.
    def delete_traces(self, experiment_id: str, max_timestamp_ms: int) -> None:
        body = {"experiment_id": experiment_id, "max_timestamp_millis": max_timestamp_ms}
        response = self.client.post(f"{self.traces_url}/delete-traces", json=body)
        response.raise_for_status()


# MLflow's trace search syntax: strings quoted, numbers bare; callers pass only checked values.
def filter_clause(key: str, comparator: str, value) -> str:
    if value is None:
        return f"{key} {comparator}"
    shown = f"'{value}'" if isinstance(value, str) else value
    return f"{key} {comparator} {shown}"


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
