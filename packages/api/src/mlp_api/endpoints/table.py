from sqlalchemy import JSON, Boolean, Column, DateTime, Integer, String, Table

from mlp_api.storage.database import metadata
from mlp_core import config

endpoint = Table(
    "endpoint",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String, nullable=False),
    # Names its URL; it stays the same across restarts.
    Column("uuid", String, nullable=False, unique=True),
    # In plain text, so users can look it up again.
    Column("key", String, nullable=False),
    Column("owner", String, nullable=False),
    # The spec with its model pinned.
    Column("spec", JSON, nullable=False),
    # Every Reference it serves from, pinned: its model, and an Adapter's base.
    Column("uses", JSON, nullable=False),
    # Every Endpoint uses one GPU; the column stays for databases from before #124.
    Column("gpus", Integer, nullable=False, default=1),
    Column("status", String, nullable=False),
    # Why it crashed, until a restart.
    Column("failure", String),
    Column("created_at", DateTime(timezone=True), nullable=False),
    # Whether the Judge scores every `judge_every`th of its Request Traces since `judging_since`,
    # at most `judge_max_per_hour`; `judge_count` counts the traces seen, so a restart goes on.
    Column("judged", Boolean, nullable=False, default=False),
    Column("judge_every", Integer, nullable=False, default=config.JUDGE_EVERY),
    Column("judge_max_per_hour", Integer, nullable=False, default=config.JUDGE_MAX_PER_HOUR),
    Column("judging_since", DateTime(timezone=True)),
    Column("judge_count", Integer, nullable=False, default=0),
)
not_stopped = endpoint.c.status != "stopped"
