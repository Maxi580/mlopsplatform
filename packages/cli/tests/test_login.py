import json
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import trustme
import yaml
from typer.testing import CliRunner

from mlp_cli.api import api_client
from mlp_cli.main import app
from mlp_core import api_paths

PASSWORD = "correct horse battery staple"
TOKEN = "issued-token"


class FakeApi(BaseHTTPRequestHandler):
    def do_POST(self):
        login = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if login["password"] == PASSWORD:
            self.answer(200, {"token": TOKEN})
        else:
            self.answer(401, {"detail": "Wrong password"})

    def do_GET(self):
        self.answer(200 if self.headers["Authorization"] == f"Bearer {TOKEN}" else 401, None)

    def answer(self, status, body):
        content = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, *args):
        pass


@pytest.fixture
def platform_ca():
    return trustme.CA()


@pytest.fixture
def platform_url(platform_ca):
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeApi)
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    platform_ca.issue_cert("127.0.0.1").configure_cert(context)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"https://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    (tmp_path / ".mlp").mkdir()
    return tmp_path


def write_profile(home, url, ca):
    ca.cert_pem.write_to_path(str(home / "ca.crt"))
    profile = {"url": url, "ca_cert": str(home / "ca.crt")}
    (home / ".mlp" / "profile.yaml").write_text(yaml.safe_dump(profile))


def mlp_login(password):
    return CliRunner().invoke(app, ["login"], input=f"{password}\n")


def test_login_stores_a_token_that_later_commands_send(home, platform_url, platform_ca):
    write_profile(home, platform_url, platform_ca)

    result = mlp_login(PASSWORD)

    assert result.exit_code == 0, result.output
    with api_client() as client:
        assert client.get(api_paths.VERIFY).status_code == 200


def test_login_with_a_wrong_password_fails_without_a_token(home, platform_url, platform_ca):
    write_profile(home, platform_url, platform_ca)

    result = mlp_login("wrong password")

    assert result.exit_code == 1
    assert "Wrong password" in result.output
    assert not (home / ".mlp" / "token").exists()


def test_login_trusts_only_the_ca_from_the_profile(home, platform_url):
    write_profile(home, platform_url, trustme.CA())

    result = mlp_login(PASSWORD)

    assert result.exit_code != 0
    assert not (home / ".mlp" / "token").exists()
