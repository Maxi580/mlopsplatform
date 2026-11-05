from sqlalchemy import create_engine, select

from mlp_api.endpoints.table import endpoint
from mlp_api.judge.settings import JudgeSettings, judge_settings, store_judge_settings
from mlp_api.storage.database import create_tables
from mlp_core import api_paths


def test_storage_shows_usage_per_bucket_against_the_object_store_size(logged_in_api, object_store):
    object_store.buckets["mlflow"]["1/run/artifacts/model/weights"] = b"x" * 300
    object_store.buckets["mlpipeline"]["logs"] = b"x" * 20

    response = logged_in_api.get(api_paths.STORAGE)

    assert response.status_code == 200, response.text
    assert response.json() == {
        "buckets": [
            {"name": "mlflow", "size_bytes": 300},
            {"name": "mlpipeline", "size_bytes": 20},
            {"name": "platform", "size_bytes": 0},
        ],
        # object_store_size from the settings ConfigMap, 100Gi in the test settings.
        "capacity_bytes": 100 * 2**30,
    }


def test_storage_requires_login(api):
    assert api.get(api_paths.STORAGE).status_code == 401


def test_an_earlier_versions_endpoint_table_gets_the_judging_columns(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'platform.db'}")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE endpoint (id INTEGER PRIMARY KEY, name VARCHAR NOT NULL,"
            " uuid VARCHAR NOT NULL, key VARCHAR NOT NULL, owner VARCHAR NOT NULL,"
            " spec JSON NOT NULL, uses JSON NOT NULL, gpus INTEGER NOT NULL,"
            " status VARCHAR NOT NULL, created_at DATETIME NOT NULL)"
        )
        connection.exec_driver_sql(
            "INSERT INTO endpoint VALUES (1, 'chat', 'u', 'k', 'shared', '{}', '[]', 1,"
            " 'running', '2026-10-01 00:00:00')"
        )

    create_tables(engine)
    create_tables(engine)

    with engine.connect() as connection:
        [row] = connection.execute(select(endpoint)).all()
    assert (row.judged, row.judge_every, row.judge_max_per_hour) == (False, 10, 60)
    assert (row.judging_since, row.judge_count) == (None, 0)


def test_an_earlier_versions_customised_rubric_survives_as_the_system_prompt(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'platform.db'}")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE judge (id INTEGER PRIMARY KEY, enabled BOOLEAN NOT NULL, spec JSON,"
            " rubric TEXT NOT NULL)"
        )
        connection.exec_driver_sql("INSERT INTO judge VALUES (1, 0, NULL, 'Be strict.')")

    create_tables(engine)
    create_tables(engine)

    assert judge_settings(engine).system_prompt == "Be strict."
    store_judge_settings(engine, JudgeSettings(system_prompt="Be kind."))
    assert judge_settings(engine).system_prompt == "Be kind."
