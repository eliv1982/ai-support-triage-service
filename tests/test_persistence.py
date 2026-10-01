import json

import pytest
from sqlalchemy import select

from app.models import Ticket

REQUEST = {
    "text": "Мне дважды списали оплату по счёту №123.",
    "channel": "email",
    "client_id": "client-persist",
}


def load_tickets(db_session) -> list[Ticket]:
    # Drop cached state so the assertions read what was actually committed.
    db_session.expire_all()
    return list(db_session.scalars(select(Ticket)))


def persisted_fields(ticket: Ticket) -> dict:
    return {
        "client_id": ticket.client_id,
        "channel": ticket.channel,
        "text": ticket.text,
        "category": ticket.category,
        "confidence": ticket.confidence,
        "escalate": ticket.escalate,
        "draft_reply": ticket.draft_reply,
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
    assert persisted_fields(ticket) == {**request, **llm_result, "error": None}
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
        "error": "LLM unavailable",
    }


@pytest.mark.parametrize(
    "content, expected_error",
    [
        pytest.param("not json", "Invalid JSON returned by LLM", id="invalid-json"),
        pytest.param(None, "Invalid JSON returned by LLM", id="empty-content"),
        pytest.param(
            '{"category": "refund", "draft_reply": "Hi", "confidence": "high", "escalate": false}',
            "LLM response did not match schema",
            id="out-of-contract-category",
        ),
        pytest.param(
            '{"category": "billing", "draft_reply": "Hi", "confidence": "high"}',
            "LLM response did not match schema",
            id="missing-escalate",
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
        "error": expected_error,
    }
