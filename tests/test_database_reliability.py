from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from app.database import Base
from app.models import Ticket
from app.repository import save_ticket
from app.schemas import TriageRequest, TriageResponse

RESULT = TriageResponse(
    category="billing", draft_reply="We will look into it.", confidence="high", escalate=False
)


def test_save_ticket_returns_its_pooled_connection_at_commit(tmp_path):
    """A request's session stays open until teardown. If save_ticket left a connection
    checked out until then (as a post-commit refresh does), requests that have already
    saved would pin the pool while others wait for it; under a saturated worker pool
    that waiting never ends (QueuePool timeouts, HTTP 500s). A pool of exactly one
    connection makes any such leak immediately visible."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'tickets.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=2,
    )
    Base.metadata.create_all(bind=engine)
    make_session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    first, second = make_session(), make_session()
    try:
        save_ticket(
            first,
            TriageRequest(text="first", channel="email", client_id="a"),
            RESULT,
            error="boom",
        )
        # `first` is deliberately still open here.
        assert engine.pool.checkedout() == 0

        # With a single pooled connection this raises QueuePool's TimeoutError if the
        # first save still holds it.
        save_ticket(second, TriageRequest(text="second", channel="chat", client_id="b"), RESULT)
    finally:
        first.close()
        second.close()

    with make_session() as check:
        rows = check.scalars(select(Ticket).order_by(Ticket.id)).all()
        stored = [
            (r.client_id, r.channel, r.text, r.category, r.confidence, r.escalate, r.draft_reply, r.error)
            for r in rows
        ]
    engine.dispose()

    assert stored == [
        ("a", "email", "first", "billing", "high", False, "We will look into it.", "boom"),
        ("b", "chat", "second", "billing", "high", False, "We will look into it.", None),
    ]
