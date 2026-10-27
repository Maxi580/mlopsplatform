import secrets

from sqlalchemy import Engine, select, update

from mlp_api.endpoints.table import endpoint, not_stopped
from mlp_core import config


def new_endpoint_key() -> str:
    return secrets.token_urlsafe(config.ENDPOINT_KEY_BYTES)


def endpoint_opened_by(engine: Engine, endpoint_uuid: str, key: str) -> str | None:
    """The name of the Endpoint with the UUID, if the key is its Endpoint Key and it runs."""
    with engine.connect() as connection:
        query = select(endpoint.c.name, endpoint.c.key).where(
            endpoint.c.uuid == endpoint_uuid, not_stopped
        )
        found = connection.execute(query).first()
    if found is None or not secrets.compare_digest(found.key.encode(), key.encode()):
        return None
    return found.name


def refresh_endpoint_key(engine: Engine, name: str) -> None:
    """Replaces the Endpoint's key; the old one stops working. LookupError if there is none."""
    with engine.begin() as connection:
        named = endpoint.c.name == name
        refreshed = connection.execute(
            update(endpoint).where(named).values(key=new_endpoint_key())
        ).rowcount
    if not refreshed:
        raise LookupError(f"No Endpoint {name}")
