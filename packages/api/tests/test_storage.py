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
