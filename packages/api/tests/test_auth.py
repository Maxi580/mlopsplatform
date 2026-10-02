import time

import jwt
import pytest
from fastapi.testclient import TestClient

from mlp_api.app import app
from mlp_api.config import JWT_ALGORITHM
from mlp_core import api_paths

from .conftest import JWT_SECRET, PASSWORD

TWELVE_HOURS = 12 * 60 * 60


def login(api, password=PASSWORD):
    return api.post(api_paths.LOGIN, json={"password": password})


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def test_login_returns_a_token_and_a_cookie_valid_for_12_hours(api):
    response = login(api)

    assert response.status_code == 200
    token = response.json()["token"]
    assert jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])["exp"] == pytest.approx(
        time.time() + TWELVE_HOURS, abs=60
    )
    cookie = response.headers["set-cookie"]
    assert f"Max-Age={TWELVE_HOURS}" in cookie
    assert "HttpOnly" in cookie
    assert "Secure" in cookie


def test_login_with_a_wrong_password_is_rejected(api):
    response = login(api, "wrong password")

    assert response.status_code == 401
    assert "set-cookie" not in response.headers


def test_login_is_rejected_before_a_password_was_set(platform_database):
    with TestClient(app, base_url="https://testserver") as client:
        assert login(client, "").status_code == 401


def test_repeated_failed_logins_are_rate_limited_even_for_the_right_password(api):
    for _ in range(5):
        assert login(api, "wrong password").status_code == 401

    assert login(api).status_code == 429


def test_verify_accepts_the_login_cookie(api):
    login(api)

    assert api.get(api_paths.VERIFY).status_code == 200


def test_verify_accepts_the_bearer_token(api):
    token = login(api).json()["token"]

    assert api.get(api_paths.VERIFY, headers=bearer(token)).status_code == 200


def test_verify_rejects_requests_without_a_token(api):
    assert api.get(api_paths.VERIFY).status_code == 401


@pytest.mark.parametrize("path", [api_paths.VERIFY, "/pipelines", "/docs"])
def test_expired_tokens_are_rejected(api, path):
    expired = jwt.encode({"exp": int(time.time()) - 1}, JWT_SECRET, algorithm=JWT_ALGORITHM)

    assert api.get(path, headers=bearer(expired)).status_code == 401


def test_tokens_signed_with_another_key_are_rejected(api):
    forged = jwt.encode(
        {"exp": int(time.time()) + 60}, "another-key-of-32-bytes-or-more!", JWT_ALGORITHM
    )

    assert api.get(api_paths.VERIFY, headers=bearer(forged)).status_code == 401


@pytest.mark.parametrize("path", ["/pipelines", "/docs", "/openapi.json"])
def test_api_routes_require_a_token(api, path):
    assert api.get(path).status_code == 401


def test_api_routes_are_reachable_with_a_token(api):
    token = login(api).json()["token"]

    assert api.get("/openapi.json", headers=bearer(token)).status_code == 200


def test_set_password_resets_the_password(api, run_set_password):
    run_set_password("a brand new password")

    assert login(api).status_code == 401
    assert login(api, "a brand new password").status_code == 200


def test_set_password_stores_no_plaintext(api, platform_database):
    assert PASSWORD.encode() not in platform_database.read_bytes()


def test_set_password_refuses_short_passwords(platform_database, run_set_password):
    with pytest.raises(SystemExit):
        run_set_password("short")
