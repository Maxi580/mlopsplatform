import json
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import trustme
import yaml

from mlp_core import api_paths

PASSWORD = "correct horse battery staple"
TOKEN = "issued-token"


class FakeApi(BaseHTTPRequestHandler):
    """Answers logins itself and other POSTs with the test's `answers`, recording every body."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.received.append((self.path, body))
        if self.path != api_paths.LOGIN:
            self.answer(*self.server.answers[self.path])
        elif body["password"] == PASSWORD:
            self.answer(200, {"token": TOKEN})
        else:
            self.answer(401, {"detail": "Wrong password"})

    def do_GET(self):
        if self.path in self.server.answers:
            self.answer(*self.server.answers[self.path])
        else:
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
def fake_api(platform_ca):
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeApi)
    server.received, server.answers = [], {}
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    platform_ca.issue_cert("127.0.0.1").configure_cert(context)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    server.url = f"https://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    (tmp_path / ".mlp").mkdir()
    return tmp_path


def write_profile(home, url, ca, **profile):
    ca.cert_pem.write_to_path(str(home / "ca.crt"))
    profile = {"url": url, "ca_cert": str(home / "ca.crt"), **profile}
    (home / ".mlp" / "profile.yaml").write_text(yaml.safe_dump(profile))
