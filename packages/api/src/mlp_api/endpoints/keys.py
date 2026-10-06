import secrets

from sqlalchemy import Engine, select, update

from mlp_api.endpoints.table import endpoint, not_stopped
from mlp_core import config


def new_endpoint_key() -> str:
    return secrets.token_urlsafe(config.ENDPOINT_KEY_BYTES)


def endpoint_key_opens(engine: Engine, endpoint_uuid: str, key: str) -> bool:
    """Whether the key is the Endpoint Key of the Endpoint with the UUID, and it isn't stopped."""
    with engine.connect() as connection:
        query = select(endpoint.c.key).where(endpoint.c.uuid == endpoint_uuid, not_stopped)
        found = connection.execute(query).scalar()
    return found is not None and secrets.compare_digest(found.encode(), key.encode())


def refresh_endpoint_key(engine: Engine, name: str) -> None:
    """Replaces the Endpoint's key; the old one stops working. LookupError if it isn't running."""
    with engine.begin() as connection:
        named = endpoint.c.name == name
        refreshed = connection.execute(
            update(endpoint).where(named, not_stopped).values(key=new_endpoint_key())
        ).rowcount
    if not refreshed:
        raise LookupError(f"No Endpoint {name} is running")
