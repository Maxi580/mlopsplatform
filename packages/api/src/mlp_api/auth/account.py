import getpass
import os
import sys

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError
from sqlalchemy import Column, Engine, Integer, String, Table, create_engine, delete, insert, select

from mlp_api.database import create_tables, metadata
from mlp_core.config import MIN_PASSWORD_LENGTH

account = Table(
    "account",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("password_hash", String, nullable=False),
)

# argon2-cffi hashes with Argon2id by default.
password_hasher = PasswordHasher()


def password_matches(engine: Engine, password: str) -> bool:
    with engine.connect() as connection:
        password_hash = connection.execute(select(account.c.password_hash)).scalar()
    if password_hash is None:
        return False
    try:
        return password_hasher.verify(password_hash, password)
    except VerificationError:
        return False


def set_password() -> None:
    """`set-password`: stores the shared account's password, read from stdin, as a hash."""
    if sys.stdin.isatty():
        password = getpass.getpass("Shared password: ")
    else:
        password = sys.stdin.readline().rstrip("\n")
    if len(password) < MIN_PASSWORD_LENGTH:
        sys.exit(f"The password needs at least {MIN_PASSWORD_LENGTH} characters")

    engine = create_engine(os.environ["DATABASE_URL"])
    create_tables(engine)
    with engine.begin() as connection:
        connection.execute(delete(account))
        connection.execute(
            insert(account).values(id=1, password_hash=password_hasher.hash(password))
        )
    engine.dispose()
