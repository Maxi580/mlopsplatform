import math
import uuid
from pathlib import Path

from sqlalchemy import JSON, Column, Engine, String, Table, delete, insert, select

from mlp_api.models.mlflow import MLflow
from mlp_api.models.registry import pin_full_weights
from mlp_api.models.upload_checks import file_list_problems, uploaded_model_tags
from mlp_api.pipelines.hugging_face import HuggingFace, pin_base_model
from mlp_api.storage.database import metadata
from mlp_api.storage.object_store import ObjectStore
from mlp_core import config

# An upload between start and complete; its files land in the artifact bucket under `prefix`.
model_upload = Table(
    "model_upload",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("base", String),
    Column("tool_parser", String),
    # File path -> the ID of its multipart upload and its size in bytes.
    Column("files", JSON, nullable=False),
)


def start_model_upload(
    engine: Engine,
    object_store: ObjectStore,
    model_registry: MLflow,
    hugging_face: HuggingFace,
    name: str,
    files: dict[str, int],
    base: str | None,
    tool_parser: str | None,
) -> dict:
    """The upload's ID and each file's part URLs; ValueError if the files can't make a version."""
    # 1. What can be checked before gigabytes travel: the file names and the base.
    problems = file_list_problems(set(files), base)
    if problems:
        raise ValueError("; ".join(problems))
    if base and base.startswith("hf:"):
        base = pin_base_model(hugging_face, base, None)
    elif base:
        base = pin_full_weights(model_registry, base)

    # 2. One multipart upload per file, under a prefix of its own.
    upload_id = uuid.uuid4().hex
    bucket, prefix = model_registry.artifact_bucket, upload_prefix(upload_id)
    multipart_uploads, answer = {}, []
    for path, size in files.items():
        multipart_id = object_store.start_multipart_upload(bucket, prefix + path)
        multipart_uploads[path] = [multipart_id, size]
        parts = max(1, math.ceil(size / config.MODEL_UPLOAD_PART_SIZE))
        urls = object_store.part_upload_urls(bucket, prefix + path, multipart_id, parts)
        answer.append({"path": path, "part_urls": urls})

    # 3. Remembered until the client completes it.
    with engine.begin() as connection:
        connection.execute(
            insert(model_upload).values(
                id=upload_id, name=name, base=base, tool_parser=tool_parser, files=multipart_uploads
            )
        )
    return {"id": upload_id, "part_size_bytes": config.MODEL_UPLOAD_PART_SIZE, "files": answer}


def complete_model_upload(
    engine: Engine, object_store: ObjectStore, model_registry: MLflow, upload_id: str
) -> dict:
    """The new Model Version; LookupError if unknown, ValueError if its files fail the checks."""
    # 1. The upload, which is only completed once.
    with engine.begin() as connection:
        upload = connection.execute(
            select(model_upload).where(model_upload.c.id == upload_id)
        ).first()
        if upload is None:
            raise LookupError(f"No model upload {upload_id}")
        connection.execute(delete(model_upload).where(model_upload.c.id == upload_id))

    # 2. Each file's parts joined, and every byte there; a part that never arrived leaves a gap.
    bucket, prefix = model_registry.artifact_bucket, upload_prefix(upload_id)
    try:
        for path, (multipart_id, _) in upload.files.items():
            object_store.complete_multipart_upload(bucket, prefix + path, multipart_id)
        arrived = {
            item["Key"].removeprefix(prefix): item["Size"]
            for item in object_store.list_objects(bucket, prefix)
        }
        short = [path for path, (_, size) in upload.files.items() if arrived.get(path) != size]
        if short:
            raise ValueError(f"{', '.join(short)} did not arrive in full; upload again")
    except ValueError:
        object_store.delete_all(bucket, prefix)
        raise

    # 3. Checked and registered.
    return register_model_files(
        object_store, model_registry, upload.name, prefix, upload.base, upload.tool_parser
    )


def upload_model_directory(
    object_store: ObjectStore, model_registry: MLflow, name: str, directory: Path
) -> dict:
    """The full-weight Model Version made of a local directory, checked like a user's upload."""
    # 1. Every file under a prefix of its own; hidden ones, like a download's .cache, aren't model.
    bucket, prefix = model_registry.artifact_bucket, upload_prefix(uuid.uuid4().hex)
    for path in directory.rglob("*"):
        relative = path.relative_to(directory).as_posix()
        if path.is_file() and not any(part.startswith(".") for part in relative.split("/")):
            object_store.upload_file(path, prefix + relative, bucket)

    # 2. Checked and registered.
    return register_model_files(object_store, model_registry, name, prefix, None, None)


def register_model_files(
    object_store: ObjectStore,
    model_registry: MLflow,
    name: str,
    prefix: str,
    base: str | None,
    tool_parser: str | None,
) -> dict:
    """The next Model Version of the files under the prefix; ValueError if rejected, files gone."""
    bucket = model_registry.artifact_bucket
    paths = {item["Key"].removeprefix(prefix) for item in object_store.list_objects(bucket, prefix)}
    try:
        tags = uploaded_model_tags(
            paths, base, tool_parser, lambda path: object_store.read(bucket, prefix + path)
        )
    except ValueError:
        object_store.delete_all(bucket, prefix)
        raise
    return {"name": name, "version": model_registry.create_model_version(name, prefix, tags)}


def upload_prefix(upload_id: str) -> str:
    return f"uploads/{upload_id}/"
