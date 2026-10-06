from urllib.parse import urlsplit

import pytest

from mlp_api.pipelines.hugging_face import HubModel
from mlp_core import api_paths

from .conftest import PASSWORD
from .test_endpoints import COMMIT, QWEN, endpoints, start, stop


@pytest.fixture
def qwen_on_the_hub(logged_in_api, hugging_face):
    hugging_face.models[QWEN] = HubModel(commit=COMMIT, needs_remote_code=False, model_type="qwen3")


def opens(api, url: str, key: str | None) -> bool:
    """Whether the forwardAuth check lets a request with the key through to the Endpoint's URL."""
    headers = {"X-Forwarded-Uri": f"{urlsplit(url).path}/models?limit=1"}
    if key is not None:
        headers["Authorization"] = f"Bearer {key}"
    response = api.get(api_paths.VERIFY_ENDPOINT_KEY, headers=headers)
    assert response.status_code in (200, 401), response.text
    return response.status_code == 200


def refresh(api, name="chat"):
    return api.post(api_paths.REFRESH_ENDPOINT_KEY.format(name=name))


def test_an_endpoint_key_opens_its_endpoint_and_nothing_else(logged_in_api, qwen_on_the_hub):
    chat = start(logged_in_api).json()
    other = start(logged_in_api, name="other").json()
    login_token = logged_in_api.post(api_paths.LOGIN, json={"password": PASSWORD}).json()["token"]

    assert opens(logged_in_api, chat["url"], chat["key"])
    assert not opens(logged_in_api, chat["url"], None)
    assert not opens(logged_in_api, chat["url"], "wrong")
    assert not opens(logged_in_api, chat["url"], other["key"])
    assert not opens(logged_in_api, chat["url"], login_token)


def test_the_key_check_needs_no_login(api, logged_in_api, qwen_on_the_hub):
    chat = start(logged_in_api).json()
    api.cookies.clear()

    assert opens(api, chat["url"], chat["key"])


def test_a_stopped_endpoints_key_opens_nothing(logged_in_api, qwen_on_the_hub):
    chat = start(logged_in_api).json()
    stop(logged_in_api)

    assert not opens(logged_in_api, chat["url"], chat["key"])


def test_a_restarted_endpoint_gets_a_new_key(logged_in_api, qwen_on_the_hub):
    first = start(logged_in_api).json()
    stop(logged_in_api)

    second = start(logged_in_api).json()

    assert first["key"] != second["key"]
    assert not opens(logged_in_api, second["url"], first["key"])


def test_a_refreshed_key_replaces_the_old_one_at_the_same_url(logged_in_api, qwen_on_the_hub):
    started = start(logged_in_api).json()

    response = refresh(logged_in_api)

    assert response.status_code == 200, response.text
    refreshed = response.json()
    assert refreshed["url"] == started["url"]
    assert refreshed["key"] != started["key"]
    assert not opens(logged_in_api, started["url"], started["key"])
    assert opens(logged_in_api, started["url"], refreshed["key"])
    assert [e["key"] for e in endpoints(logged_in_api)] == [refreshed["key"]]


def test_only_a_running_endpoints_key_can_be_refreshed(logged_in_api, qwen_on_the_hub):
    start(logged_in_api)
    stop(logged_in_api)

    assert refresh(logged_in_api).status_code == 404
    assert refresh(logged_in_api, "unknown").status_code == 404


def test_refreshing_a_key_requires_login(api, cluster):
    assert refresh(api).status_code == 401


def test_the_route_lets_only_the_endpoint_key_through(logged_in_api, qwen_on_the_hub, cluster):
    start(logged_in_api)

    [route] = cluster.endpoints["chat"]["route"]["spec"]["routes"]
    assert [m["name"] for m in route["middlewares"]] == ["endpoint-key", "endpoint-strip-prefix"]
