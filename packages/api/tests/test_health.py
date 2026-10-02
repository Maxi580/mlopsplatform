from fastapi.testclient import TestClient

from mlp_api.app import app


def test_health_answers_once_the_database_is_migrated(
    settings_configmap_env, tmp_path, monkeypatch
):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'platform.db'}")

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
