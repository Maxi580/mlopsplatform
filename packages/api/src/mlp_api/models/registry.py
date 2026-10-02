from sqlalchemy import Engine

from mlp_api.models.mlflow import MLflow
from mlp_api.pipelines.lifecycle import refuse_while_in_use
from mlp_api.storage.object_store import ObjectStore
from mlp_core.pipeline_request.references import model_reference


def list_models(model_registry: MLflow, object_store: ObjectStore) -> list[dict]:
    """Every Registered Model with its versions, oldest first, with sizes and lineage tags."""
    models = {}
    for found in sorted(model_registry.model_versions(), key=lambda v: (v.name, v.version)):
        size = object_store.size_of(model_registry.artifact_bucket, found.artifact_prefix)
        models.setdefault(found.name, []).append(
            {"version": found.version, "size_bytes": size, "tags": found.tags}
        )
    return [{"name": name, "versions": versions} for name, versions in models.items()]


def delete_model_version(
    engine: Engine, model_registry: MLflow, object_store: ObjectStore, name: str, version: int
) -> None:
    """Deletes the version and its files; LookupError if unknown, ValueError while in use."""
    # 1. The version, as the registry knows it.
    found = next(
        (v for v in model_registry.model_versions() if (v.name, v.version) == (name, version)),
        None,
    )
    if found is None:
        raise LookupError(f"Registered Model {name} has no version {version}")

    # 2. Refused while an unfinished Pipeline uses it or is still producing it.
    refuse_while_in_use(
        engine, model_reference(name, version), produced_by=found.tags.get("pipeline")
    )

    # 3. The registry entry, then its files.
    model_registry.delete_model_version(name, version)
    object_store.delete_all(model_registry.artifact_bucket, found.artifact_prefix)
