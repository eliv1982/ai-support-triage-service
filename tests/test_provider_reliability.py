"""Timeout, retry and client-reuse policy, observed through the real OpenAI SDK transport.

Everything talks to a loopback stub, so the network guard stays satisfied and no real
provider is ever contacted. The stub's `attempts` counter is the ground truth for how
many requests the SDK actually sent.
"""

import socket
import threading
import time

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.database import get_db
from app.dependencies import get_rate_limiter, get_triage_service
from app.main import app
from app.schemas import TriageRequest
from app.services import llm_client, triage_service
from app.services.triage_service import TriageService

PAYLOAD = TriageRequest(text="I was charged twice", channel="email", client_id="client-1")


def real_service(base_url: str, monkeypatch: pytest.MonkeyPatch, timeout: float = 10.0):
    """The production client factory, fed settings that point at a loopback address."""
    settings = Settings(
        _env_file=None, openai_base_url=base_url, openai_timeout_seconds=timeout
    )
    monkeypatch.setattr(llm_client, "get_settings", lambda: settings)
    return TriageService(settings=settings, llm_client=llm_client.build_openai_client())


# --- the policy is explicit ---------------------------------------------------------


def test_timeout_defaults_to_ten_seconds():
    assert Settings(_env_file=None).openai_timeout_seconds == 10.0


def test_timeout_can_be_set_from_the_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "2.5")
    assert Settings(_env_file=None).openai_timeout_seconds == 2.5


@pytest.mark.parametrize("bad", ["0", "-1", "inf", "nan", "soon"])
def test_timeout_must_be_a_positive_finite_number(
    monkeypatch: pytest.MonkeyPatch, bad: str
):
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", bad)
    with pytest.raises(ValidationError, match="openai_timeout_seconds"):
        Settings(_env_file=None)


def test_client_is_built_with_the_configured_timeout_and_no_retries(
    monkeypatch: pytest.MonkeyPatch,
):
    settings = Settings(_env_file=None, openai_timeout_seconds=7.5)
    monkeypatch.setattr(llm_client, "get_settings", lambda: settings)

    client = llm_client.build_openai_client()

    # Not the SDK defaults (a 600 s read timeout and 2 retries).
    assert client.timeout == 7.5
    assert client.max_retries == 0
    assert llm_client.OPENAI_MAX_RETRIES == 0


# --- failures make exactly one upstream attempt -------------------------------------


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503])
def test_retryable_upstream_errors_are_attempted_once_then_fail_safe(
    status: int, provider_stub, monkeypatch, expected_failsafe
):
    # These are all statuses the SDK retries by default (3 attempts in total).
    provider_stub.status = status
    service = real_service(provider_stub.base_url, monkeypatch)

    result = service.triage(PAYLOAD)

    assert provider_stub.attempts == 1
    assert result.used_fallback is True
    assert result.response.model_dump() == expected_failsafe
    assert str(status) in result.error


def test_retry_after_header_does_not_hold_a_worker_thread(
    provider_stub, monkeypatch, expected_failsafe
):
    provider_stub.status = 429
    provider_stub.retry_after = 30  # the SDK default would sleep this long, then retry
    service = real_service(provider_stub.base_url, monkeypatch)

    started = time.monotonic()
    result = service.triage(PAYLOAD)
    elapsed = time.monotonic() - started

    assert provider_stub.attempts == 1
    assert result.response.model_dump() == expected_failsafe
    assert elapsed < 5  # generous: the point is "not 30 s", not an exact figure


def test_connection_failures_are_not_retried_either(monkeypatch, expected_failsafe):
    """A server that accepts and immediately drops each connection counts every try."""
    accepted = []
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    listener.settimeout(0.2)
    stop = threading.Event()

    def accept_and_drop():
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except OSError:
                continue
            accepted.append(1)
            conn.close()

    thread = threading.Thread(target=accept_and_drop, daemon=True)
    thread.start()
    try:
        port = listener.getsockname()[1]
        service = real_service(f"http://127.0.0.1:{port}/v1", monkeypatch)
        result = service.triage(PAYLOAD)
    finally:
        stop.set()
        thread.join(timeout=5)
        listener.close()

    assert len(accepted) == 1
    assert result.used_fallback is True
    assert result.response.model_dump() == expected_failsafe


# --- a stalled provider is cut off ---------------------------------------------------


def test_stalled_provider_fails_within_the_configured_timeout(
    provider_stub, monkeypatch, expected_failsafe
):
    provider_stub.delay = 60  # never answers in time; released when the test ends
    service = real_service(provider_stub.base_url, monkeypatch, timeout=0.5)

    started = time.monotonic()
    result = service.triage(PAYLOAD)
    elapsed = time.monotonic() - started

    assert provider_stub.attempts == 1  # a timeout is not retried
    assert result.used_fallback is True
    assert result.response.model_dump() == expected_failsafe
    assert "timed out" in result.error.lower()
    # Bounds are loose on purpose: it waited for the timeout, and nowhere near the SDK's
    # 600 s default (or a retry on top of it).
    assert 0.4 <= elapsed < 5


# --- the client is built once -------------------------------------------------------


def test_triage_service_is_process_scoped(monkeypatch, fake_llm):
    monkeypatch.setattr(triage_service, "build_openai_client", lambda: fake_llm(content="{}"))
    get_triage_service.cache_clear()
    try:
        assert get_triage_service() is get_triage_service()
    finally:
        get_triage_service.cache_clear()


def test_one_sdk_client_serves_every_http_request(
    provider_stub, monkeypatch, db_session, rate_limiter
):
    settings = Settings(_env_file=None, openai_base_url=provider_stub.base_url)
    monkeypatch.setattr(llm_client, "get_settings", lambda: settings)

    built = []
    real_openai = llm_client.OpenAI

    class CountingOpenAI(real_openai):
        def __init__(self, *args, **kwargs):
            built.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(llm_client, "OpenAI", CountingOpenAI)
    # conftest makes the factory refuse inside triage_service; give it the real one back
    # (still pointed at loopback) so the app's own dependency wiring is what gets tested.
    monkeypatch.setattr(triage_service, "build_openai_client", llm_client.build_openai_client)

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_rate_limiter] = lambda: rate_limiter
    get_triage_service.cache_clear()
    try:
        http = TestClient(app)
        for i in range(5):
            response = http.post(
                "/triage", json={"text": "hi", "channel": "chat", "client_id": f"c{i}"}
            )
            assert response.status_code == 200
            assert response.json()["category"] == "billing"  # the stub's answer, not a fallback
    finally:
        app.dependency_overrides.clear()
        get_triage_service.cache_clear()

    assert provider_stub.attempts == 5  # every request really went through the SDK
    assert len(built) == 1
