from sqlalchemy import Engine, MetaData

# Every domain module adds its tables here.
metadata = MetaData()


# Creates missing tables only; a schema change in v1 means recreating the database.
def create_tables(engine: Engine) -> None:
    metadata.create_all(engine)
