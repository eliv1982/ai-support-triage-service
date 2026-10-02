"""Schema creation, and the in-place upgrade of a database created before `used_fallback`.

There is no migration framework (see database.init_db). Every database here is a scratch
file under tmp_path; nothing touches a developer's real app.db.
"""

import sqlite3

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.database import init_db
from app.main import app
from app.models import Ticket
from app.repository import save_ticket
from app.schemas import TriageRequest, TriageResponse

# The `tickets` table exactly as it was created before Stage 3B (the schema of the demo
# app.db that existing users have on disk), including its two indexes.
LEGACY_DDL = """
CREATE TABLE tickets (
    id INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
    client_id VARCHAR(255) NOT NULL,
    channel VARCHAR(20) NOT NULL,
    text TEXT NOT NULL,
    category VARCHAR(20) NOT NULL,
    confidence VARCHAR(20) NOT NULL,
    escalate BOOLEAN NOT NULL,
    draft_reply TEXT NOT NULL,
    error TEXT,
    PRIMARY KEY (id)
);
CREATE INDEX ix_tickets_id ON tickets (id);
CREATE INDEX ix_tickets_client_id ON tickets (client_id);
"""

# (client_id, category, confidence, escalate, error): what the old code could have stored.
LEGACY_ROWS = [
    ("c-ok", "billing", "high", 0, None),
    ("c-provider", "other", "low", 1, "Error code: 401 - Incorrect API key provided: sk-..."),
    ("c-limited", "other", "low", 1, "Rate limit exceeded"),
]

RESULT = TriageResponse(
    category="support", draft_reply="On it.", confidence="high", escalate=False
)


@pytest.fixture
def make_engine(tmp_path):
    engines = []

    def _make(name: str = "tickets.db"):
        engine = create_engine(
            f"sqlite:///{tmp_path / name}", connect_args={"check_same_thread": False}
        )
        engines.append(engine)
        return engine

    yield _make
    for engine in engines:
        engine.dispose()


def build_legacy_db(path) -> None:
    con = sqlite3.connect(path)
    con.executescript(LEGACY_DDL)
    for client_id, category, confidence, escalate, error in LEGACY_ROWS:
        con.execute(
            "INSERT INTO tickets (client_id, channel, text, category, confidence, escalate,"
            " draft_reply, error) VALUES (?, 'email', 'legacy text', ?, ?, ?, 'legacy reply', ?)",
            (client_id, category, confidence, escalate, error),
        )
    con.commit()
    con.close()


def read(path, sql: str) -> list[tuple]:
    con = sqlite3.connect(path)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def column_names(path) -> list[str]:
    return [row[1] for row in read(path, "PRAGMA table_info(tickets)")]


# --- fresh database -------------------------------------------------------------------


def test_fresh_database_is_created_with_the_discriminator_column(make_engine, tmp_path):
    engine = make_engine()

    init_db(engine)

    assert "used_fallback" in column_names(tmp_path / "tickets.db")


def test_the_column_is_not_null_and_defaults_to_not_a_fallback(make_engine, tmp_path):
    engine = make_engine()
    init_db(engine)
    path = tmp_path / "tickets.db"

    # PRAGMA table_info rows are (cid, name, type, notnull, dflt_value, pk).
    info = {row[1]: row for row in read(path, "PRAGMA table_info(tickets)")}
    assert info["used_fallback"][3] == 1

    con = sqlite3.connect(path)
    con.execute(
        "INSERT INTO tickets (client_id, channel, text, category, confidence, escalate,"
        " draft_reply) VALUES ('c', 'chat', 't', 'other', 'low', 1, 'r')"
    )
    con.commit()
    con.close()
    assert read(path, "SELECT used_fallback FROM tickets") == [(0,)]


def test_init_db_is_idempotent_on_a_fresh_database(make_engine, tmp_path):
    engine = make_engine()

    init_db(engine)
    init_db(engine)

    assert column_names(tmp_path / "tickets.db").count("used_fallback") == 1


# --- a database created before the column existed ------------------------------------


def test_legacy_database_is_upgraded_without_losing_anything(make_engine, tmp_path):
    path = tmp_path / "tickets.db"
    build_legacy_db(path)
    before = read(path, "SELECT id, client_id, text, error, created_at FROM tickets ORDER BY id")
    assert "used_fallback" not in column_names(path)

    init_db(make_engine())

    assert "used_fallback" in column_names(path)
    after = read(path, "SELECT id, client_id, text, error, created_at FROM tickets ORDER BY id")
    assert after == before  # every pre-existing row, byte for byte
    index_names = {r[0] for r in read(path, "SELECT name FROM sqlite_master WHERE type='index'")}
    assert {"ix_tickets_id", "ix_tickets_client_id"} <= index_names


def test_legacy_rows_are_backfilled_from_their_error_column(make_engine, tmp_path):
    """In the old schema a non-null error was only ever written next to a fail-safe
    result, so it is a faithful marker of what used_fallback should have said."""
    path = tmp_path / "tickets.db"
    build_legacy_db(path)

    init_db(make_engine())

    assert read(path, "SELECT client_id, used_fallback FROM tickets ORDER BY id") == [
        ("c-ok", 0),
        ("c-provider", 1),
        ("c-limited", 1),
    ]


def test_an_upgraded_database_accepts_new_tickets(make_engine, tmp_path):
    path = tmp_path / "tickets.db"
    build_legacy_db(path)
    engine = make_engine()
    init_db(engine)

    with sessionmaker(bind=engine)() as db:
        save_ticket(db, TriageRequest(text="new", channel="chat", client_id="n1"), RESULT)
        save_ticket(
            db,
            TriageRequest(text="new", channel="chat", client_id="n2"),
            RESULT,
            error="provider_error",
            used_fallback=True,
        )
        rows = db.execute(
            select(Ticket.client_id, Ticket.used_fallback, Ticket.error).order_by(Ticket.id)
        ).all()

    assert [tuple(r) for r in rows][-2:] == [
        ("n1", False, None),
        ("n2", True, "provider_error"),
    ]
    assert len(rows) == len(LEGACY_ROWS) + 2


def test_the_upgrade_runs_once_and_does_not_rewrite_later_rows(make_engine, tmp_path):
    path = tmp_path / "tickets.db"
    build_legacy_db(path)
    engine = make_engine()
    init_db(engine)
    con = sqlite3.connect(path)
    # A modern row where the flag and the error legitimately differ from the legacy rule.
    con.execute(
        "INSERT INTO tickets (client_id, channel, text, category, confidence, escalate,"
        " draft_reply, error, used_fallback) VALUES ('c-new', 'chat', 't', 'other', 'low',"
        " 1, 'r', 'provider_error', 0)"
    )
    con.commit()
    con.close()

    init_db(engine)  # e.g. the next app start

    assert read(path, "SELECT used_fallback FROM tickets WHERE client_id = 'c-new'") == [(0,)]


# --- wiring ---------------------------------------------------------------------------


def test_application_startup_upgrades_an_existing_database_file(
    make_engine, tmp_path, monkeypatch
):
    path = tmp_path / "tickets.db"
    build_legacy_db(path)
    monkeypatch.setattr("app.main.engine", make_engine())

    with TestClient(app):  # entering the context runs the lifespan startup
        pass

    assert "used_fallback" in column_names(path)
    assert len(read(path, "SELECT id FROM tickets")) == len(LEGACY_ROWS)


def test_application_startup_creates_a_fresh_database(make_engine, tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.engine", make_engine("brand-new.db"))

    with TestClient(app):
        pass

    assert "used_fallback" in column_names(tmp_path / "brand-new.db")
