import os
import socket
from collections.abc import Generator
from types import SimpleNamespace

# Settings rejects a missing/placeholder key at import time, so a fake one must be in
# place before any app module loads. Overwritten (not setdefault) on purpose: a real key
# exported in the developer's shell must never reach the test process.
os.environ["OPENAI_API_KEY"] = "test-key-not-a-real-credential"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.database import Base, get_db
from app.dependencies import get_rate_limiter, get_triage_service
from app.main import app
from app.rate_limiter import InMemoryRateLimiter
from app.services.triage_service import TriageService
from provider_stub import ProviderStub

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(autouse=True)
def block_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if a test tries to reach a real host (e.g. the LLM provider).

    Loopback stays open: the event loop behind TestClient needs it on some platforms.
    """
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def check(sock: socket.socket, address) -> None:
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            if address[0] not in LOOPBACK_HOSTS:
                raise RuntimeError(f"Network access blocked in tests: {address!r}")

    def guarded_connect(self, address):
        check(self, address)
        return real_connect(self, address)

    def guarded_connect_ex(self, address):
        check(self, address)
        return real_connect_ex(self, address)

    def refuse_real_client():
        raise RuntimeError("Tests must inject a fake llm_client, not build the real one")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(
        "app.services.triage_service.build_openai_client", refuse_real_client
    )


class FakeLLMClient:
    """Stands in for the OpenAI SDK client and records every request it receives."""

    def __init__(self, content: str | None = None, error: Exception | None = None):
        self.calls: list[dict] = []
        self._content = content
        self._error = error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture
def provider_stub(monkeypatch: pytest.MonkeyPatch):
    """A loopback Chat Completions server: set .status/.delay/.retry_after, read .attempts."""
    # A developer's proxy settings must not divert loopback traffic.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    stub = ProviderStub()
    try:
        yield stub
    finally:
        stub.close()


@pytest.fixture
def fake_llm():
    """Factory: fake_llm(content=...) or fake_llm(error=...)."""
    return FakeLLMClient


@pytest.fixture
def make_service():
    """Factory: make_service(fake_client, model=...) -> TriageService."""

    def _make(llm_client, model: str = "test-model") -> TriageService:
        # _env_file=None keeps the developer's local .env out of the tests.
        settings = Settings(_env_file=None, openai_model=model)
        return TriageService(settings=settings, llm_client=llm_client)

    return _make


@pytest.fixture
def expected_failsafe() -> dict:
    """The fail-safe response, written out literally on purpose.

    Do not derive this from fallback_response(): comparing the code with itself would
    not notice if the fail-safe silently stopped escalating.
    """
    return {
        "category": "other",
        "draft_reply": (
            "Thank you for your message. This request needs review by a member of our "
            "support team before it can be answered."
        ),
        "confidence": "low",
        "escalate": True,
    }


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def rate_limiter() -> InMemoryRateLimiter:
    return InMemoryRateLimiter(limit_per_minute=2)


@pytest.fixture
def client(db_session: Session, rate_limiter: InMemoryRateLimiter):
    def _build(mock_service):
        def override_get_db():
            try:
                yield db_session
            finally:
                pass

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_triage_service] = lambda: mock_service
        app.dependency_overrides[get_rate_limiter] = lambda: rate_limiter
        return TestClient(app)

    yield _build
    app.dependency_overrides.clear()
