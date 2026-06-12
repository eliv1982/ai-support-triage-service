from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.dependencies import get_rate_limiter, get_triage_service
from app.main import app
from app.rate_limiter import InMemoryRateLimiter


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
