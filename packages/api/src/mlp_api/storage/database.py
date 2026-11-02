from sqlalchemy import Engine, MetaData, inspect, literal
from sqlalchemy.schema import Column

# Every domain module adds its tables here.
metadata = MetaData()


# Other schema changes in v1 mean recreating the database.
def create_tables(engine: Engine) -> None:
    """Creates missing tables, and adds the columns an earlier version's tables lack."""
    metadata.create_all(engine)
    inspector = inspect(engine)
    with engine.begin() as connection:
        for table in metadata.sorted_tables:
            existing = {column["name"] for column in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name not in existing:
                    connection.exec_driver_sql(
                        f"ALTER TABLE {table.name} ADD COLUMN {column_definition(column, engine)}"
                    )


# A new column on existing rows needs its default to be NOT NULL.
def column_definition(column: Column, engine: Engine) -> str:
    definition = f"{column.name} {column.type.compile(engine.dialect)}"
    if column.default is None:
        return definition
    default = literal(column.default.arg).compile(
        dialect=engine.dialect, compile_kwargs={"literal_binds": True}
    )
    return f"{definition} DEFAULT {default} NOT NULL"
