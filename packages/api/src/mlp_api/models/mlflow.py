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
    """The MLflow Model Registry, which holds every Registered Model; tests swap in a fake."""

    def __init__(self):
        self.artifact_bucket = os.environ["MLFLOW_BUCKET"]
        self.client = httpx.Client(base_url=f"{os.environ['MLFLOW_URL']}/api/2.0/mlflow")

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


# MLflow proxies artifacts: `mlflow-artifacts:/<path>` lives at `<path>` in its bucket.
def model_version(found: dict) -> ModelVersion:
    return ModelVersion(
        name=found["name"],
        version=int(found["version"]),
        tags={tag["key"]: tag["value"] for tag in found.get("tags", [])},
        artifact_prefix=found["source"].removeprefix("mlflow-artifacts:/").rstrip("/") + "/",
    )
