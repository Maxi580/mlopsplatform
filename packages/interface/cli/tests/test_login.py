import trustme
from typer.testing import CliRunner

from mlp_cli.api import api_client
from mlp_cli.main import app
from mlp_core import api_paths

from .conftest import PASSWORD, write_profile


def mlp_login(password):
    return CliRunner().invoke(app, ["login"], input=f"{password}\n")


def test_login_stores_a_token_that_later_commands_send(home, fake_api, platform_ca):
    write_profile(home, fake_api.url, platform_ca)

    result = mlp_login(PASSWORD)

    assert result.exit_code == 0, result.output
    with api_client() as client:
        assert client.get(api_paths.VERIFY).status_code == 200


def test_login_with_a_wrong_password_fails_without_a_token(home, fake_api, platform_ca):
    write_profile(home, fake_api.url, platform_ca)

    result = mlp_login("wrong password")

    assert result.exit_code == 1
    assert "Wrong password" in result.output
    assert not (home / ".mlp" / "token").exists()


def test_login_trusts_only_the_ca_from_the_profile(home, fake_api):
    write_profile(home, fake_api.url, trustme.CA())

    result = mlp_login(PASSWORD)

    assert result.exit_code != 0
    assert not (home / ".mlp" / "token").exists()
