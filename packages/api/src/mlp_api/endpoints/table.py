from sqlalchemy import JSON, Column, DateTime, Integer, String, Table

from mlp_api.storage.database import metadata

endpoint = Table(
    "endpoint",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String, nullable=False),
    # Names its URL; a restart is a new Endpoint with a new one.
    Column("uuid", String, nullable=False, unique=True),
    # In plain text, so users can look it up again.
    Column("key", String, nullable=False),
    Column("owner", String, nullable=False),
    # The spec with its model pinned.
    Column("spec", JSON, nullable=False),
    # Every Reference it serves from, pinned: its model, and an Adapter's base.
    Column("uses", JSON, nullable=False),
    Column("gpus", Integer, nullable=False),
    Column("status", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
not_stopped = endpoint.c.status != "stopped"
