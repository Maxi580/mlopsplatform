from mlp_core import api_paths

READING = {"memory_total_bytes": 128 * 2**30, "shared_with_system": True}


def reserved(api) -> dict:
    response = api.get(api_paths.GPU_RESERVED)
    assert response.status_code == 200, response.text
    return response.json()


def test_the_fullest_machine_and_the_recommended_maximum_are_read(logged_in_api, cluster):
    cluster.gpu_readings = {"host": [READING], "joined": [READING]}
    cluster.gpu_slices = {"host": 3, "joined": 12}

    assert reserved(logged_in_api) == {"reserved_percent": 60, "recommended_max": 0.85}


def test_a_machine_without_jobs_has_nothing_reserved(logged_in_api, cluster):
    cluster.gpu_readings = {"host": [READING]}

    assert reserved(logged_in_api)["reserved_percent"] == 0


def test_whole_gpus_count_of_all_a_machines_gpus_while_sharing_is_off(logged_in_api, cluster):
    logged_in_api.app.state.settings.gpu_sharing = False
    cluster.gpu_readings = {"host": [READING, READING]}
    cluster.gpu_slices = {"host": 1}

    assert reserved(logged_in_api)["reserved_percent"] == 50


def test_nothing_is_said_if_no_machine_answers(logged_in_api, cluster):
    cluster.gpu_readings = {"empty": []}

    assert reserved(logged_in_api)["reserved_percent"] is None


def test_nothing_is_said_if_the_cluster_cant_be_read(logged_in_api, cluster, monkeypatch):
    def fail():
        raise ConnectionError("the Kubernetes API is down")

    monkeypatch.setattr(cluster, "machine_gpu_slices", fail)

    assert reserved(logged_in_api)["reserved_percent"] is None


def test_what_is_reserved_requires_login(api):
    assert api.get(api_paths.GPU_RESERVED).status_code == 401
