"""Request boundary: what POST /triage accepts, and that a rejected request has no effects.

A 422 must happen before anything else: no provider call, no ticket row, and no spent
rate-limit quota.
"""

import json
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app.models import Ticket

GOOD_OUTPUT = json.dumps(
    {
        "category": "billing",
        "draft_reply": "Looking into it.",
        "confidence": "high",
        "escalate": False,
    }
)


@pytest.fixture
def api(client, db_session, fake_llm, make_service):
    llm = fake_llm(content=GOOD_OUTPUT)
    return SimpleNamespace(http=client(make_service(llm)), llm=llm, db=db_session)


def post(api, **overrides):
    body = {
        "text": "I need help with my invoice.",
        "channel": "email",
        "client_id": "client-1",
        **overrides,
    }
    return api.http.post("/triage", json=body)


def tickets(api) -> list[Ticket]:
    api.db.expire_all()
    return list(api.db.scalars(select(Ticket)))


def sent_to_provider(api) -> dict:
    return json.loads(api.llm.calls[0]["messages"][1]["content"])


def assert_rejected_without_side_effects(api, response) -> None:
    assert response.status_code == 422
    assert tickets(api) == []
    assert api.llm.calls == []


# --- A. ticket text ------------------------------------------------------------------

BLANK_TEXTS = [
    pytest.param("", id="empty"),
    pytest.param(" ", id="one-space"),
    pytest.param("     ", id="spaces"),
    pytest.param("\n", id="newline"),
    pytest.param(" \t\r\n ", id="ascii-whitespace-mix"),
    pytest.param(" ", id="no-break-space"),
    pytest.param("　 ", id="unicode-spaces"),
]


@pytest.mark.parametrize("text", BLANK_TEXTS)
def test_blank_ticket_text_is_rejected(api, text):
    assert_rejected_without_side_effects(api, post(api, text=text))


@pytest.mark.parametrize("body", [{"channel": "chat", "client_id": "c"}, {"text": None}])
def test_missing_or_null_ticket_text_is_rejected(api, body):
    response = api.http.post("/triage", json={"channel": "chat", "client_id": "c", **body})

    assert_rejected_without_side_effects(api, response)


def test_surrounding_whitespace_is_trimmed_before_it_is_stored_or_sent(api):
    # Deliberate: padding is not content. The stored ticket and the provider see the same
    # trimmed text, so the two can never disagree about what the customer wrote.
    response = post(api, text="  \n Hello there \t ")

    assert response.status_code == 200
    (ticket,) = tickets(api)
    assert ticket.text == "Hello there"
    assert sent_to_provider(api)["text"] == "Hello there"


def test_inner_whitespace_is_left_alone(api):
    text = "first line\n\n  second   line\twith a tab"

    assert post(api, text=text).status_code == 200

    (ticket,) = tickets(api)
    assert ticket.text == text
    assert sent_to_provider(api)["text"] == text


def test_ticket_text_may_be_exactly_the_maximum_length(api):
    assert post(api, text="a" * 2000).status_code == 200


def test_ticket_text_over_the_maximum_length_is_rejected(api):
    assert_rejected_without_side_effects(api, post(api, text="a" * 2001))


def test_the_length_limit_applies_to_the_trimmed_text(api):
    assert post(api, text="  " + "a" * 2000 + "  ").status_code == 200

    (ticket,) = tickets(api)
    assert len(ticket.text) == 2000


# --- B. client_id --------------------------------------------------------------------

REALISTIC_CLIENT_IDS = [
    pytest.param("client-123", id="slug"),
    pytest.param("client_123", id="underscore"),
    pytest.param("7", id="single-digit"),
    pytest.param("user@example.com", id="email"),
    pytest.param("user+billing@example.co.uk", id="email-with-tag"),
    pytest.param("9f1c2e4a-7b1d-4c55-8a0e-1f2e3d4c5b6a", id="uuid"),
    pytest.param("01HZX8Q4V6N2K9M3T5R7W1Y0AB", id="ulid"),
    pytest.param("acme:user.42", id="namespaced"),
    pytest.param("192.168.0.1", id="ipv4"),
    pytest.param("2001:db8::1", id="ipv6"),
    pytest.param("клиент-1", id="non-ascii-letters"),
    pytest.param("x" * 128, id="exactly-128-chars"),
]


@pytest.mark.parametrize("client_id", REALISTIC_CLIENT_IDS)
def test_realistic_client_ids_are_accepted_and_stored_as_given(api, client_id):
    response = post(api, client_id=client_id)

    assert response.status_code == 200
    (ticket,) = tickets(api)
    assert ticket.client_id == client_id


UNSAFE_CLIENT_IDS = [
    pytest.param("", id="empty"),
    pytest.param(" ", id="one-space"),
    pytest.param("   ", id="spaces"),
    pytest.param("\t", id="tab"),
    pytest.param("\n", id="newline-only"),
    pytest.param("a b", id="inner-space"),
    pytest.param(" a", id="leading-space"),
    pytest.param("a ", id="trailing-space"),
    pytest.param("a\nb", id="newline"),
    pytest.param("a\rb", id="carriage-return"),
    pytest.param("a\r\nb", id="crlf"),
    pytest.param("a\tb", id="tab-inside"),
    pytest.param("a\x00b", id="nul"),
    pytest.param("a\x1b[31mb", id="ansi-escape"),
    pytest.param("a\x7fb", id="delete"),
    pytest.param("a\u0085b", id="next-line"),
    pytest.param("a b", id="line-separator"),
    pytest.param("a b", id="paragraph-separator"),
    pytest.param("a‮b", id="bidi-override"),
    pytest.param("a​b", id="zero-width-space"),
    pytest.param("a b", id="no-break-space"),
    pytest.param("x" * 129, id="129-chars"),
    pytest.param("x" * 10_000, id="10k-chars"),
]


@pytest.mark.parametrize("client_id", UNSAFE_CLIENT_IDS)
def test_unsafe_or_oversized_client_id_is_rejected(api, client_id):
    assert_rejected_without_side_effects(api, post(api, client_id=client_id))


@pytest.mark.parametrize("body", [{"text": "hi", "channel": "chat"}, {"client_id": None}])
def test_missing_or_null_client_id_is_rejected(api, body):
    response = api.http.post("/triage", json={"text": "hi", "channel": "chat", **body})

    assert_rejected_without_side_effects(api, response)


# --- C. a rejected request is free ---------------------------------------------------


def test_rejected_requests_do_not_consume_rate_limit_quota(api):
    # The fixture limiter allows 2 requests per client.
    for _ in range(5):
        assert post(api, text="   ").status_code == 422

    assert post(api).status_code == 200
    assert post(api).status_code == 200
    assert post(api).status_code == 429
    assert len(tickets(api)) == 2
