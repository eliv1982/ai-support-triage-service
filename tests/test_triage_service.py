import json
from typing import get_args

import pytest

from app.schemas import Category, Confidence, TriageRequest, TriageResponse
from app.services.triage_service import fallback_response

INVALID_JSON = "Invalid JSON returned by LLM"
SCHEMA_MISMATCH = "LLM response did not match schema"

PAYLOAD = TriageRequest(
    text="I was charged twice for invoice #123.",
    channel="email",
    client_id="client-42",
)


def valid_output(**overrides) -> dict:
    data = {
        "category": "billing",
        "draft_reply": "We are reviewing the duplicate charge.",
        "confidence": "medium",
        "escalate": False,
    }
    data.update(overrides)
    return data


def without(key: str) -> dict:
    data = valid_output()
    del data[key]
    return data


def assert_failsafe(result, expected_failsafe: dict, error: str) -> None:
    assert result.used_fallback is True
    assert result.error == error
    assert result.response.model_dump() == expected_failsafe
    # The critical contract, stated on its own: a failure never becomes "no escalation".
    assert result.response.escalate is True


# --- A. Literal fail-safe -------------------------------------------------------


def test_fallback_response_is_the_literal_escalating_failsafe(expected_failsafe):
    assert fallback_response().model_dump() == expected_failsafe


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(RuntimeError("LLM unavailable"), id="runtime-error"),
        pytest.param(TimeoutError("request timed out"), id="timeout"),
        pytest.param(ConnectionError("connection refused"), id="connection"),
    ],
)
def test_provider_failure_returns_escalating_failsafe(
    fake_llm, make_service, expected_failsafe, error
):
    service = make_service(fake_llm(error=error))

    result = service.triage(PAYLOAD)

    assert_failsafe(result, expected_failsafe, error=str(error))


# --- B. Malformed / unusable LLM output ----------------------------------------
# D. These cases double as the guard for local schema validation: if validation of
# the LLM result were removed or bypassed, out-of-contract output would be returned
# to the caller as a normal (non-fallback, possibly non-escalated) triage result.

UNUSABLE_OUTPUTS = [
    pytest.param("this is not json", INVALID_JSON, id="not-json"),
    pytest.param('{"category": "billing", ', INVALID_JSON, id="truncated-json"),
    pytest.param("", INVALID_JSON, id="empty-string"),
    pytest.param("   \n", INVALID_JSON, id="whitespace-only"),
    pytest.param(None, INVALID_JSON, id="null-content"),
    pytest.param("[]", SCHEMA_MISMATCH, id="json-array"),
    pytest.param("null", SCHEMA_MISMATCH, id="json-null"),
    pytest.param("{}", SCHEMA_MISMATCH, id="empty-object"),
    pytest.param(json.dumps(without("category")), SCHEMA_MISMATCH, id="missing-category"),
    pytest.param(json.dumps(without("draft_reply")), SCHEMA_MISMATCH, id="missing-draft"),
    pytest.param(json.dumps(without("confidence")), SCHEMA_MISMATCH, id="missing-confidence"),
    pytest.param(json.dumps(without("escalate")), SCHEMA_MISMATCH, id="missing-escalate"),
    pytest.param(
        json.dumps(valid_output(category="refund")), SCHEMA_MISMATCH, id="invalid-category"
    ),
    pytest.param(
        json.dumps(valid_output(confidence="certain")),
        SCHEMA_MISMATCH,
        id="invalid-confidence",
    ),
    pytest.param(
        json.dumps(valid_output(confidence=0.9)), SCHEMA_MISMATCH, id="numeric-confidence"
    ),
    pytest.param(
        json.dumps(valid_output(escalate="maybe")), SCHEMA_MISMATCH, id="non-boolean-escalate"
    ),
    pytest.param(
        json.dumps(valid_output(escalate=None)), SCHEMA_MISMATCH, id="null-escalate"
    ),
    pytest.param(
        json.dumps(valid_output(draft_reply="")), SCHEMA_MISMATCH, id="empty-draft-reply"
    ),
]


@pytest.mark.parametrize("content, expected_error", UNUSABLE_OUTPUTS)
def test_unusable_llm_output_returns_escalating_failsafe(
    fake_llm, make_service, expected_failsafe, content, expected_error
):
    service = make_service(fake_llm(content=content))

    result = service.triage(PAYLOAD)

    assert_failsafe(result, expected_failsafe, error=expected_error)


@pytest.mark.parametrize("escalate", [False, True])
def test_valid_llm_output_is_passed_through_without_fallback(
    fake_llm, make_service, escalate
):
    output = valid_output(
        category="complaint", confidence="high", escalate=escalate
    )
    service = make_service(fake_llm(content=json.dumps(output)))

    result = service.triage(PAYLOAD)

    assert result.used_fallback is False
    assert result.error is None
    assert result.response == TriageResponse(**output)
    assert result.response.model_dump() == output


# --- E. Outbound LLM request contract ------------------------------------------


def test_request_forwards_configured_model_and_requests_json_mode(fake_llm, make_service):
    llm = fake_llm(content=json.dumps(valid_output()))
    service = make_service(llm, model="model-from-settings")

    service.triage(PAYLOAD)

    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["model"] == "model-from-settings"
    assert call["response_format"] == {"type": "json_object"}
    assert call["temperature"] == 0.2


def test_request_sends_system_prompt_then_ticket_as_user_message(fake_llm, make_service):
    llm = fake_llm(content=json.dumps(valid_output()))
    service = make_service(llm)
    payload = TriageRequest(
        text="Мне дважды списали оплату по счёту №123.",
        channel="chat",
        client_id="client-ru",
    )

    service.triage(payload)

    system_message, user_message = llm.calls[0]["messages"]
    assert system_message["role"] == "system"
    assert user_message["role"] == "user"
    # client_id is deliberately not asserted: whether it should reach the provider is
    # a separate, later decision and this suite must not fight it.
    sent = json.loads(user_message["content"])
    assert sent["text"] == payload.text
    assert sent["channel"] == "chat"


def test_system_prompt_states_the_response_contract_the_schema_enforces(fake_llm, make_service):
    llm = fake_llm(content=json.dumps(valid_output()))
    service = make_service(llm)

    service.triage(PAYLOAD)

    prompt = llm.calls[0]["messages"][0]["content"]
    required_terms = (
        *TriageResponse.model_fields,
        *get_args(Category),
        *get_args(Confidence),
    )
    missing = [term for term in required_terms if term not in prompt]
    assert missing == []
    assert "JSON" in prompt
