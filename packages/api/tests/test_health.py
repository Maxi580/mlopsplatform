def test_health_answers_without_login_once_the_database_is_migrated(api):
    response = api.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
