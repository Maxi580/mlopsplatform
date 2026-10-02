from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine


def upgrade_database(engine: Engine) -> None:
    config = Config()
    config.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
