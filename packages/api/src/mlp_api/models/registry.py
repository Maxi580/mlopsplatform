from sqlalchemy import Engine

from mlp_api.in_use.users import refuse_while_in_use
from mlp_api.models.mlflow import MLflow, ModelVersion
from mlp_api.storage.object_store import ObjectStore
from mlp_core.pipeline_request.references import model_reference, split_model_reference


def list_models(model_registry: MLflow, object_store: ObjectStore) -> list[dict]:
    """Every Registered Model with its versions, oldest first, with sizes and lineage tags."""
    models = {}
    for found in sorted(model_registry.model_versions(), key=lambda v: (v.name, v.version)):
        size = object_store.size_of(model_registry.artifact_bucket, found.artifact_prefix)
        models.setdefault(found.name, []).append(
            {"version": found.version, "size_bytes": size, "tags": found.tags}
        )
    return [{"name": name, "versions": versions} for name, versions in models.items()]


def find_model_version(
    model_registry: MLflow, name: str, version: int | None = None
) -> ModelVersion | None:
    """The version if it exists, the latest one when none is named, else None."""
    found = [
        v
        for v in model_registry.model_versions()
        if v.name == name and version in (None, v.version)
    ]
    return max(found, key=lambda v: v.version, default=None)


def pin_full_weights(model_registry: MLflow, reference: str) -> str:
    """The `model:` Reference pinned to a full-weight version; ValueError saying why it can't be."""
    found = find_referenced_model_version(model_registry, reference)
    pinned = model_reference(found.name, found.version)
    if found.tags.get("weights") == "adapter":
        raise ValueError(f"{pinned} is an Adapter; only full weights can be built on")
    return pinned


def find_referenced_model_version(model_registry: MLflow, reference: str) -> ModelVersion:
    """The version a `model:` Reference names, else the latest; ValueError if there is none."""
    name, version = split_model_reference(reference)
    found = find_model_version(model_registry, name, version)
    if found is None and version is None:
        raise ValueError(f"no Registered Model `{name}`")
    if found is None:
        raise ValueError(f"`{name}` has no version {version}")
    return found


def model_version_files(
    model_registry: MLflow, object_store: ObjectStore, name: str, version: int
) -> list[dict]:
    """Each file of the version with its size and a presigned URL; LookupError if unknown."""
    found = find_model_version(model_registry, name, version)
    if found is None:
        raise LookupError(f"Registered Model {name} has no version {version}")
    bucket = model_registry.artifact_bucket
    return [
        {
            "path": item["Key"].removeprefix(found.artifact_prefix),
            "size_bytes": item["Size"],
            "url": object_store.download_url(item["Key"], bucket),
        }
        for item in object_store.list_objects(bucket, found.artifact_prefix)
    ]


def delete_model_version(
    engine: Engine, model_registry: MLflow, object_store: ObjectStore, name: str, version: int
) -> None:
    """Deletes the version and its files; LookupError if unknown, ValueError while in use."""
    # 1. The version, as the registry knows it.
    found = find_model_version(model_registry, name, version)
    if found is None:
        raise LookupError(f"Registered Model {name} has no version {version}")

    # 2. Refused while an unfinished Pipeline uses it or is still producing it.
    refuse_while_in_use(
        engine, model_reference(name, version), produced_by=found.tags.get("pipeline")
    )

    # 3. The registry entry, then its files.
    model_registry.delete_model_version(name, version)
    object_store.delete_all(model_registry.artifact_bucket, found.artifact_prefix)
