import json

import pytest
from sqlalchemy import func, select

from app.models import Ticket

REQUEST = {
    "text": "Мне дважды списали оплату по счёту №123.",
    "channel": "email",
    "client_id": "client-persist",
}

GOOD_OUTPUT = json.dumps(
    {
        "category": "billing",
        "draft_reply": "Looking into it.",
        "confidence": "high",
        "escalate": False,
    }
)


def load_tickets(db_session) -> list[Ticket]:
    # Drop cached state so the assertions read what was actually committed.
    db_session.expire_all()
    return list(db_session.scalars(select(Ticket)))


def ticket_count(db_session) -> int:
    db_session.expire_all()
    return db_session.scalar(select(func.count()).select_from(Ticket))


def persisted_fields(ticket: Ticket) -> dict:
    return {
        "client_id": ticket.client_id,
        "channel": ticket.channel,
        "text": ticket.text,
        "category": ticket.category,
        "confidence": ticket.confidence,
        "escalate": ticket.escalate,
        "draft_reply": ticket.draft_reply,
        "used_fallback": ticket.used_fallback,
        "error": ticket.error,
    }


# Every channel is exercised so a hard-coded channel cannot pass by coincidence.
@pytest.mark.parametrize("channel", ["email", "form", "chat"])
@pytest.mark.parametrize("escalate", [False, True])
def test_successful_triage_persists_request_and_llm_result(
    client, db_session, fake_llm, make_service, escalate, channel
):
    request = {**REQUEST, "channel": channel}
    llm_result = {
        "category": "billing",
        "draft_reply": "We are reviewing the duplicate charge.",
        "confidence": "medium",
        "escalate": escalate,
    }
    test_client = client(make_service(fake_llm(content=json.dumps(llm_result))))

    response = test_client.post("/triage", json=request)

    assert response.status_code == 200
    assert response.json() == llm_result
    (ticket,) = load_tickets(db_session)
    assert persisted_fields(ticket) == {
        **request,
        **llm_result,
        "used_fallback": False,
        "error": None,
    }
    assert ticket.id is not None
    assert ticket.created_at is not None


def test_provider_failure_persists_failsafe_ticket_with_error(
    client, db_session, fake_llm, make_service, expected_failsafe
):
    test_client = client(make_service(fake_llm(error=RuntimeError("LLM unavailable"))))

    response = test_client.post("/triage", json=REQUEST)

    assert response.status_code == 200
    assert response.json() == expected_failsafe
    (ticket,) = load_tickets(db_session)
    assert persisted_fields(ticket) == {
        **REQUEST,
        **expected_failsafe,
        "used_fallback": True,
        "error": "provider_error",
    }


def test_provider_timeout_is_persisted_as_its_own_reason(
    client, db_session, fake_llm, make_service
):
    test_client = client(make_service(fake_llm(error=TimeoutError("took too long"))))

    assert test_client.post("/triage", json=REQUEST).status_code == 200

    (ticket,) = load_tickets(db_session)
    assert (ticket.used_fallback, ticket.error) == (True, "provider_timeout")


@pytest.mark.parametrize(
    "content, expected_error",
    [
        pytest.param("not json", "invalid_llm_json", id="invalid-json"),
        pytest.param(None, "invalid_llm_json", id="empty-content"),
        pytest.param(
            '{"category": "refund", "draft_reply": "Hi", "confidence": "high", "escalate": false}',
            "invalid_llm_schema",
            id="out-of-contract-category",
        ),
        pytest.param(
            '{"category": "billing", "draft_reply": "Hi", "confidence": "high"}',
            "invalid_llm_schema",
            id="missing-escalate",
        ),
        pytest.param(
            '{"category": "billing", "draft_reply": "Hi", "confidence": "high", "escalate": "false"}',
            "invalid_llm_schema",
            id="string-escalate",
        ),
        pytest.param(
            '{"category": "billing", "draft_reply": "Hi", "confidence": "high", "escalate": false, "note": "x"}',
            "invalid_llm_schema",
            id="extra-key",
        ),
    ],
)
def test_unusable_llm_output_persists_failsafe_ticket_with_error(
    client, db_session, fake_llm, make_service, expected_failsafe, content, expected_error
):
    test_client = client(make_service(fake_llm(content=content)))

    response = test_client.post("/triage", json=REQUEST)

    assert response.status_code == 200
    assert response.json() == expected_failsafe
    (ticket,) = load_tickets(db_session)
    assert persisted_fields(ticket) == {
        **REQUEST,
        **expected_failsafe,
        "used_fallback": True,
        "error": expected_error,
    }


def test_provider_exception_text_is_never_persisted_or_returned(
    client, db_session, fake_llm, make_service
):
    leaky = (
        "Error code: 401 - Incorrect API key provided: sk-proj-SECRET1234. "
        "Request id req_abc123 for https://api.example.test/v1"
    )
    test_client = client(make_service(fake_llm(error=RuntimeError(leaky))))

    response = test_client.post("/triage", json=REQUEST)

    assert response.status_code == 200
    assert "SECRET" not in response.text
    assert "req_abc123" not in response.text
    (ticket,) = load_tickets(db_session)
    assert ticket.error == "provider_error"
    stored = " ".join(str(value) for value in persisted_fields(ticket).values())
    assert "SECRET" not in stored
    assert "req_abc123" not in stored


def test_model_result_identical_to_the_failsafe_is_not_marked_as_fallback(
    client, db_session, fake_llm, make_service, expected_failsafe
):
    """The reason the flag exists: content alone cannot tell a fail-safe from a model
    that genuinely answered "other / low / escalate" with the very same wording."""
    test_client = client(make_service(fake_llm(content=json.dumps(expected_failsafe))))

    response = test_client.post("/triage", json=REQUEST)

    assert response.status_code == 200
    assert response.json() == expected_failsafe
    (ticket,) = load_tickets(db_session)
    assert (ticket.used_fallback, ticket.error) == (False, None)


def test_success_and_fallback_tickets_are_distinguishable_in_the_database(
    client, db_session, fake_llm, make_service
):
    ok = client(make_service(fake_llm(content=GOOD_OUTPUT)))
    assert ok.post("/triage", json={**REQUEST, "client_id": "ok-1"}).status_code == 200
    broken = client(make_service(fake_llm(error=RuntimeError("down"))))
    assert broken.post("/triage", json={**REQUEST, "client_id": "bad-1"}).status_code == 200
    garbled = client(make_service(fake_llm(content="{")))
    assert garbled.post("/triage", json={**REQUEST, "client_id": "bad-2"}).status_code == 200

    db_session.expire_all()
    fallback_clients = set(
        db_session.scalars(select(Ticket.client_id).where(Ticket.used_fallback))
    )
    normal_clients = set(
        db_session.scalars(select(Ticket.client_id).where(Ticket.used_fallback.is_(False)))
    )

    assert fallback_clients == {"bad-1", "bad-2"}
    assert normal_clients == {"ok-1"}


# --- requests rejected before triage are not tickets ---------------------------------


def test_rate_limited_request_creates_no_ticket_and_never_reaches_the_provider(
    client, db_session, fake_llm, make_service
):
    llm = fake_llm(content=GOOD_OUTPUT)
    test_client = client(make_service(llm))  # the fixture limiter allows 2 per client

    accepted = [test_client.post("/triage", json=REQUEST) for _ in range(2)]
    assert [r.status_code for r in accepted] == [200, 200]
    assert ticket_count(db_session) == 2
    assert len(llm.calls) == 2

    rejected = test_client.post("/triage", json=REQUEST)

    assert rejected.status_code == 429
    assert rejected.json() == {"detail": "Rate limit exceeded"}
    assert ticket_count(db_session) == 2  # unchanged: no synthetic row
    assert len(llm.calls) == 2  # the provider was not invoked for the rejected request
    # And the two accepted ones are still ordinary, untouched tickets.
    tickets = load_tickets(db_session)
    assert [t.error for t in tickets] == [None, None]
    assert [t.used_fallback for t in tickets] == [False, False]


def test_a_client_that_is_already_over_the_limit_leaves_the_database_empty(
    client, db_session, rate_limiter, fake_llm, make_service
):
    llm = fake_llm(content=GOOD_OUTPUT)
    test_client = client(make_service(llm))
    for _ in range(2):
        assert rate_limiter.allow(REQUEST["client_id"])  # quota used up elsewhere

    response = test_client.post("/triage", json=REQUEST)

    assert response.status_code == 429
    assert ticket_count(db_session) == 0
    assert llm.calls == []


def test_repeated_rejections_never_add_rows(client, db_session, fake_llm, make_service):
    llm = fake_llm(content=GOOD_OUTPUT)
    test_client = client(make_service(llm))
    for _ in range(2):
        assert test_client.post("/triage", json=REQUEST).status_code == 200

    statuses = [test_client.post("/triage", json=REQUEST).status_code for _ in range(10)]

    assert statuses == [429] * 10
    assert ticket_count(db_session) == 2
    assert len(llm.calls) == 2


def test_rate_limiting_one_client_does_not_affect_another(
    client, db_session, fake_llm, make_service
):
    test_client = client(make_service(fake_llm(content=GOOD_OUTPUT)))
    for _ in range(3):
        test_client.post("/triage", json=REQUEST)

    other = test_client.post("/triage", json={**REQUEST, "client_id": "someone-else"})

    assert other.status_code == 200
    assert ticket_count(db_session) == 3  # 2 accepted + 1 other client; the 429 was not stored
