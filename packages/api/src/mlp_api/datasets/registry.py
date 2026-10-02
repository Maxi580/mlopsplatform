from pathlib import Path

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Engine,
    Integer,
    Row,
    String,
    Table,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError

from mlp_api.datasets.rows import row_format_of_file
from mlp_api.in_use.users import refuse_while_in_use
from mlp_api.storage.database import metadata
from mlp_api.storage.object_store import ObjectStore
from mlp_core.pipeline_request.references import dataset_key, dataset_reference

dataset_version = Table(
    "dataset_version",
    metadata,
    Column("name", String, primary_key=True),
    Column("version", Integer, primary_key=True),
    Column("row_format", String, nullable=False),
    Column("size_bytes", BigInteger, nullable=False),
    # Deleted versions keep their row, so their number is never handed out again.
    Column("deleted", Boolean, nullable=False, default=False),
)


def upload_dataset_version(
    engine: Engine, object_store: ObjectStore, name: str, path: Path
) -> dict:
    """The new Dataset Version holding the JSONL file; ValueError naming the first bad row."""
    # 1. Every row in one supported row format.
    row_format = row_format_of_file(path)

    with engine.begin() as connection:
        # 2. The next version number, counting deleted versions too.
        last = select(func.max(dataset_version.c.version)).where(dataset_version.c.name == name)
        version = (connection.execute(last).scalar() or 0) + 1
        new_version = {
            "version": version,
            "size_bytes": path.stat().st_size,
            "row_format": row_format,
        }
        try:
            connection.execute(insert(dataset_version).values(name=name, **new_version))
        except IntegrityError:
            raise ValueError("another upload took the same version number; upload again") from None
        # 3. The file, inside the transaction so a failed upload leaves no version behind.
        object_store.upload_file(path, dataset_key(name, version))
    return new_version


def list_datasets(engine: Engine) -> list[dict]:
    """Every Dataset with its versions, oldest first."""
    query = (
        select(dataset_version)
        .where(~dataset_version.c.deleted)
        .order_by(dataset_version.c.name, dataset_version.c.version)
    )
    datasets = {}
    with engine.connect() as connection:
        for row in connection.execute(query):
            datasets.setdefault(row.name, []).append(
                {"version": row.version, "size_bytes": row.size_bytes, "row_format": row.row_format}
            )
    return [{"name": name, "versions": versions} for name, versions in datasets.items()]


def find_dataset_version(engine: Engine, name: str, version: int | None = None) -> Row | None:
    """The version's row if it exists, the latest one when none is named, else None."""
    query = (
        select(dataset_version)
        .where(dataset_version.c.name == name, ~dataset_version.c.deleted)
        .order_by(dataset_version.c.version.desc())
        .limit(1)
    )
    if version is not None:
        query = query.where(dataset_version.c.version == version)
    with engine.connect() as connection:
        return connection.execute(query).first()


def delete_dataset_version(
    engine: Engine, object_store: ObjectStore, name: str, version: int
) -> None:
    """Deletes the version and its file; LookupError if unknown, ValueError while in use."""
    # 1. Refused while an unfinished Pipeline trains on it.
    refuse_while_in_use(engine, dataset_reference(name, version))

    # 2. The row is kept, the file goes.
    with engine.begin() as connection:
        deleted = connection.execute(
            update(dataset_version)
            .where(
                dataset_version.c.name == name,
                dataset_version.c.version == version,
                ~dataset_version.c.deleted,
            )
            .values(deleted=True)
        )
        if deleted.rowcount == 0:
            raise LookupError(f"Dataset {name} has no version {version}")
        object_store.delete(dataset_key(name, version))
