import asyncio
import threading

import anyio.to_thread
import httpx2
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.dependencies import get_rate_limiter, get_triage_service
from app.main import app
from app.rate_limiter import InMemoryRateLimiter
from app.schemas import TriageResponse
from app.services.triage_service import TriageResult

# More requests than the default worker-thread limit (40) that sync endpoints share.
BLOCKED_REQUESTS = 45


def test_health_response_is_exactly_the_documented_one():
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


class BlockedService:
    """Holds the calling worker thread until the test opens the gate."""

    def __init__(self, gate: threading.Event) -> None:
        self.gate = gate
        self.lock = threading.Lock()
        self.blocked = 0  # worker threads currently parked inside triage()

    def triage(self, payload) -> TriageResult:
        with self.lock:
            self.blocked += 1
        self.gate.wait(timeout=30)
        return TriageResult(
            response=TriageResponse(
                category="other", draft_reply="ok", confidence="low", escalate=True
            )
        )


def test_health_answers_while_every_sync_worker_is_busy_with_triage(tmp_path):
    """/health is liveness only; it must not queue for a worker thread behind triage work."""
    gate = threading.Event()
    engine = create_engine(
        f"sqlite:///{tmp_path / 'health.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    make_session = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def override_get_db():
        db = make_session()
        try:
            yield db
        finally:
            db.close()

    limiter = InMemoryRateLimiter(limit_per_minute=1000)
    service = BlockedService(gate)
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_rate_limiter] = lambda: limiter
    app.dependency_overrides[get_triage_service] = lambda: service

    async def scenario():
        transport = httpx2.ASGITransport(app=app)
        async with httpx2.AsyncClient(transport=transport, base_url="http://test") as http:
            posts = [
                asyncio.create_task(
                    http.post(
                        "/triage",
                        json={"text": "hi", "channel": "chat", "client_id": f"c{i}"},
                    )
                )
                for i in range(BLOCKED_REQUESTS)
            ]
            try:
                # Sustained saturation: every worker thread is parked inside triage().
                # (Merely seeing all tokens borrowed is not enough; short sync dependencies
                # of 45 simultaneous requests can briefly borrow them all.)
                workers = anyio.to_thread.current_default_thread_limiter()
                for _ in range(500):  # up to ~5 s for the worker pool to fill
                    if service.blocked >= workers.total_tokens:
                        break
                    await asyncio.sleep(0.01)
                else:
                    raise AssertionError("worker pool never filled; scenario is not valid")

                health = await asyncio.wait_for(http.get("/health"), timeout=2)
            finally:
                gate.set()
            return health, await asyncio.gather(*posts)

    try:
        health, triage_responses = asyncio.run(scenario())
    finally:
        gate.set()
        app.dependency_overrides.clear()
        engine.dispose()

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert [r.status_code for r in triage_responses] == [200] * BLOCKED_REQUESTS
