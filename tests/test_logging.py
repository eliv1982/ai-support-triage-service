"""What the application logs on the triage paths: never ticket text, never secrets, and
never a line an attacker can forge through `client_id`."""

import json
import logging

import httpx2
import openai
import pytest

TICKET_TEXT = "My card 4111 1111 1111 1111 was charged twice, call me on +1 555 0100"
TEST_API_KEY = "test-key-not-a-real-credential"  # set in conftest
LEAKED_SECRET = "sk-proj-SECRET1234"

GOOD_OUTPUT = {
    "category": "billing",
    "draft_reply": "Looking into it.",
    "confidence": "high",
    "escalate": False,
}


@pytest.fixture(autouse=True)
def capture_logs(caplog):
    caplog.set_level(logging.INFO)


def post(test_client, **overrides):
    body = {"text": TICKET_TEXT, "channel": "email", "client_id": "client-1", **overrides}
    return test_client.post("/triage", json=body)


def app_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name.startswith("app")]


def assert_nothing_sensitive(caplog) -> None:
    for fragment in (TICKET_TEXT, "4111", "555 0100", TEST_API_KEY, LEAKED_SECRET):
        assert fragment not in caplog.text, f"{fragment!r} was logged"


# --- provider failures: the category is the routine log -------------------------------


def test_provider_failure_logs_the_category_not_the_exception_text(
    client, fake_llm, make_service, caplog
):
    leaky = f"Incorrect API key provided: {LEAKED_SECRET}. Ticket was: {TICKET_TEXT}"
    test_client = client(make_service(fake_llm(error=RuntimeError(leaky))))

    assert post(test_client).status_code == 200

    assert "provider_error" in caplog.text
    assert "RuntimeError" in caplog.text  # the type is what makes it diagnosable
    assert_nothing_sensitive(caplog)
    assert all(r.exc_info is None for r in app_records(caplog))  # no traceback to leak from


def test_provider_status_code_is_logged_when_the_sdk_reports_one(
    client, fake_llm, make_service, caplog
):
    request = httpx2.Request("POST", "http://127.0.0.1/v1/chat/completions")
    error = openai.AuthenticationError(
        f"Incorrect API key provided: {LEAKED_SECRET}",
        response=httpx2.Response(401, request=request),
        body=None,
    )
    test_client = client(make_service(fake_llm(error=error)))

    assert post(test_client).status_code == 200

    assert "AuthenticationError" in caplog.text
    assert "401" in caplog.text
    assert_nothing_sensitive(caplog)


def test_provider_timeout_logs_its_own_category(client, fake_llm, make_service, caplog):
    test_client = client(make_service(fake_llm(error=TimeoutError("took too long"))))

    assert post(test_client).status_code == 200

    assert "provider_timeout" in caplog.text


# --- unusable model output: its content is derived from the ticket --------------------


@pytest.mark.parametrize(
    "content, reason",
    [
        pytest.param(f"not json {TICKET_TEXT}", "invalid_llm_json", id="not-json"),
        pytest.param(
            json.dumps({**GOOD_OUTPUT, "category": TICKET_TEXT}),
            "invalid_llm_schema",
            id="ticket-text-in-an-invalid-field",
        ),
        pytest.param(
            json.dumps({**GOOD_OUTPUT, "echo": TICKET_TEXT}),
            "invalid_llm_schema",
            id="ticket-text-in-an-extra-field",
        ),
    ],
)
def test_unusable_model_output_is_logged_without_its_content(
    client, fake_llm, make_service, caplog, content, reason
):
    test_client = client(make_service(fake_llm(content=content)))

    assert post(test_client).status_code == 200

    assert reason in caplog.text
    # Pydantic's own error text quotes the offending input, so it must not be logged.
    assert_nothing_sensitive(caplog)
    assert all(r.exc_info is None for r in app_records(caplog))


# --- the other paths ------------------------------------------------------------------


def test_a_normal_request_logs_no_ticket_text(client, fake_llm, make_service, caplog):
    test_client = client(make_service(fake_llm(content=json.dumps(GOOD_OUTPUT))))

    assert post(test_client).status_code == 200

    assert app_records(caplog)  # something is logged, and it is safe
    assert_nothing_sensitive(caplog)


def test_a_rate_limited_request_is_logged_without_ticket_text(
    client, fake_llm, make_service, caplog
):
    test_client = client(make_service(fake_llm(content=json.dumps(GOOD_OUTPUT))))
    for _ in range(2):
        post(test_client)
    caplog.clear()

    assert post(test_client).status_code == 429

    assert "Rate limit exceeded" in caplog.text
    assert "client-1" in caplog.text
    assert_nothing_sensitive(caplog)


@pytest.mark.parametrize(
    "client_id",
    [
        "x\n2026-01-01 00:00:00,000 CRITICAL app.main forged-entry",
        "x\r\nforged-entry",
        "x forged-entry",
        "x\x1b[2Kforged-entry",
    ],
)
def test_a_client_id_cannot_forge_log_records(
    client, fake_llm, make_service, caplog, client_id
):
    test_client = client(make_service(fake_llm(content=json.dumps(GOOD_OUTPUT))))

    response = post(test_client, client_id=client_id)

    assert response.status_code == 422  # rejected at the boundary, so never logged at all
    assert "forged-entry" not in caplog.text


@pytest.mark.parametrize(
    "error, content",
    [
        pytest.param(None, json.dumps(GOOD_OUTPUT), id="success"),
        pytest.param(RuntimeError("boom\nsecond line"), None, id="provider-failure"),
        pytest.param(None, "garbage\nsecond line", id="invalid-output"),
    ],
)
def test_every_log_record_is_a_single_line(client, fake_llm, make_service, caplog, error, content):
    test_client = client(make_service(fake_llm(content=content, error=error)))
    for _ in range(3):  # the third request is rate limited
        post(test_client, client_id="client-ok.1:x")

    records = app_records(caplog)
    assert records
    assert all("\n" not in record.getMessage() for record in records)
