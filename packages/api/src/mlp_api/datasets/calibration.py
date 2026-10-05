import gzip
import os
import tempfile
from pathlib import Path

from sqlalchemy import Engine, create_engine

from mlp_api.datasets.registry import find_dataset_version, upload_dataset_version
from mlp_api.pipelines.hugging_face import HuggingFace
from mlp_api.storage.database import create_tables
from mlp_api.storage.object_store import ObjectStore
from mlp_core import config


def register_default_calibration_dataset() -> None:
    """`register-calibration-dataset`: install.sh's call, with the API's own storage."""
    engine = create_engine(os.environ["DATABASE_URL"])
    create_tables(engine)
    register_calibration_dataset(engine, ObjectStore(), HuggingFace())
    engine.dispose()


def register_calibration_dataset(
    engine: Engine, object_store: ObjectStore, hugging_face: HuggingFace
) -> None:
    """Registers the default calibration Dataset's rows from Hugging Face, unless it exists."""
    if find_dataset_version(engine, config.CALIBRATION_DATASET) is not None:
        return
    source = config.CALIBRATION_SOURCE
    rows = hugging_face.dataset_file(source["repo"], source["commit"], source["file"])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rows.jsonl"
        path.write_bytes(gzip.decompress(rows))
        upload_dataset_version(engine, object_store, config.CALIBRATION_DATASET, path)
