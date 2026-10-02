import getpass
import os
import sys

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError
from sqlalchemy import Engine, create_engine, text

from mlp_api.config import MIN_PASSWORD_LENGTH
from mlp_api.database import upgrade_database

# argon2-cffi hashes with Argon2id by default.
password_hasher = PasswordHasher()


def password_matches(engine: Engine, password: str) -> bool:
    with engine.connect() as connection:
        password_hash = connection.execute(text("SELECT password_hash FROM account")).scalar()
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
    upgrade_database(engine)
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM account"))
        connection.execute(
            text("INSERT INTO account (id, password_hash) VALUES (1, :password_hash)"),
            {"password_hash": password_hasher.hash(password)},
        )
    engine.dispose()
