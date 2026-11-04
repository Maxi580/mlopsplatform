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


def gpu_reading(index: int = 0) -> dict:
    return {
        "index": index,
        "name": "NVIDIA GB10",
        "memory_total_bytes": 128 * 2**30,
        "memory_used_bytes": 32 * 2**30,
        "memory_free_bytes": 96 * 2**30,
        "shared_with_system": True,
        "utilization_percent": None,
    }


def test_gpus_show_each_machines_readings_and_the_shares_reserved_on_it(logged_in_api, cluster):
    cluster.gpu_readings = {"host": [gpu_reading()], "joined": [gpu_reading()]}
    cluster.gpu_slices = {"host": 12, "joined": 3}

    response = logged_in_api.get(api_paths.STORAGE_GPUS)

    assert response.status_code == 200, response.text
    assert response.json() == {
        "machines": [
            {"name": "host", "reserved_percent": 60, "gpus": [gpu_reading()]},
            {"name": "joined", "reserved_percent": 15, "gpus": [gpu_reading()]},
        ]
    }


def test_a_machine_without_jobs_has_nothing_reserved(logged_in_api, cluster):
    cluster.gpu_readings = {"host": [gpu_reading()]}

    [machine] = logged_in_api.get(api_paths.STORAGE_GPUS).json()["machines"]

    assert machine["reserved_percent"] == 0


def test_whole_gpus_are_reserved_of_all_a_machines_gpus_while_sharing_is_off(
    logged_in_api, cluster
):
    logged_in_api.app.state.settings.gpu_sharing = False
    cluster.gpu_readings = {"host": [gpu_reading(0), gpu_reading(1)]}
    cluster.gpu_slices = {"host": 1}

    [machine] = logged_in_api.get(api_paths.STORAGE_GPUS).json()["machines"]

    assert machine["reserved_percent"] == 50


def test_a_machine_reporting_no_gpus_is_left_out(logged_in_api, cluster):
    cluster.gpu_readings = {"empty": [], "host": [gpu_reading()]}

    machines = logged_in_api.get(api_paths.STORAGE_GPUS).json()["machines"]

    assert [machine["name"] for machine in machines] == ["host"]


def test_gpus_require_login(api):
    assert api.get(api_paths.STORAGE_GPUS).status_code == 401


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
