import logging

from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


settings = get_settings()

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False}
    if settings.database_url.startswith("sqlite")
    else {},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db(bind: Engine) -> None:
    """Create missing tables and bring a database made by an older version up to date.

    There is no migration framework: this is a one-table demo, and `create_all` never
    alters a table that already exists. Without the step below, a database file from
    before `used_fallback` existed would fail every insert, after the (paid) provider call
    had already been made. The upgrade is additive only; it deletes and rewrites nothing.
    """
    # Imported here, not at the top: models imports Base from this module. Importing it
    # registers the tickets table on Base.metadata, which create_all needs.
    from app import models  # noqa: F401

    Base.metadata.create_all(bind=bind)
    _add_used_fallback_column(bind)


def _add_used_fallback_column(bind: Engine) -> None:
    if any(c["name"] == "used_fallback" for c in inspect(bind).get_columns("tickets")):
        return

    with bind.begin() as conn:
        conn.execute(
            text("ALTER TABLE tickets ADD COLUMN used_fallback BOOLEAN NOT NULL DEFAULT FALSE")
        )
        # Before this column existed, `error` was only ever set next to a fail-safe
        # result, so it says which old rows were fallbacks.
        conn.execute(text("UPDATE tickets SET used_fallback = TRUE WHERE error IS NOT NULL"))
    logger.info("Upgraded the tickets table: added the used_fallback column")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
