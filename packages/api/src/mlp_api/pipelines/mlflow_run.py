import json

from mlp_api.models.mlflow import MLflow
from mlp_api.storage.object_store import ObjectStore
from mlp_core import config


def log_pipeline_request(
    model_registry: MLflow, object_store: ObjectStore, mlflow_run_url: str, request: dict
) -> None:
    """Writes the resolved request at the root of the Pipeline's MLflow Run; LookupError if none."""
    key = pipeline_request_key(model_registry, mlflow_run_url)
    if key is None:
        raise LookupError(f"MLflow has no Run at {mlflow_run_url} yet")
    content = json.dumps(request, indent=1).encode()
    object_store.write(key, content, model_registry.artifact_bucket)


# The Run's other artifacts, metrics and Model Versions stay.
def delete_pipeline_request(
    model_registry: MLflow, object_store: ObjectStore, mlflow_run_url: str
) -> None:
    key = pipeline_request_key(model_registry, mlflow_run_url)
    if key is not None:
        object_store.delete(key, model_registry.artifact_bucket)


def pipeline_request_key(model_registry: MLflow, mlflow_run_url: str) -> str | None:
    """Where the request lies in the artifact bucket, or None if the Run is unknown."""
    found = config.MLFLOW_RUN_ID_IN_URL.search(mlflow_run_url)
    prefix = found and model_registry.run_artifact_prefix(found[1])
    return prefix and prefix + config.PIPELINE_REQUEST_FILE
