import json
from typing import get_args

import httpx2
import openai
import pytest

from app.schemas import Category, Confidence, LLMTriageOutput, TriageRequest, TriageResponse
from app.services.triage_service import FailureReason, fallback_response

INVALID_JSON = FailureReason.INVALID_LLM_JSON
SCHEMA_MISMATCH = FailureReason.INVALID_LLM_SCHEMA
PROVIDER_ERROR = FailureReason.PROVIDER_ERROR
PROVIDER_TIMEOUT = FailureReason.PROVIDER_TIMEOUT

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


SDK_REQUEST = httpx2.Request("POST", "http://127.0.0.1/v1/chat/completions")

# The stored reason is a category, never the exception's own text (see FailureReason).
PROVIDER_FAILURES = [
    pytest.param(RuntimeError("LLM unavailable"), PROVIDER_ERROR, id="runtime-error"),
    pytest.param(ConnectionError("connection refused"), PROVIDER_ERROR, id="connection"),
    pytest.param(TimeoutError("request timed out"), PROVIDER_TIMEOUT, id="timeout"),
    pytest.param(
        openai.APITimeoutError(request=SDK_REQUEST), PROVIDER_TIMEOUT, id="sdk-timeout"
    ),
    pytest.param(
        openai.APIConnectionError(request=SDK_REQUEST), PROVIDER_ERROR, id="sdk-connection"
    ),
    pytest.param(
        openai.AuthenticationError(
            "Incorrect API key provided",
            response=httpx2.Response(401, request=SDK_REQUEST),
            body=None,
        ),
        PROVIDER_ERROR,
        id="sdk-status-error",
    ),
]


@pytest.mark.parametrize("error, reason", PROVIDER_FAILURES)
def test_provider_failure_returns_escalating_failsafe(
    fake_llm, make_service, expected_failsafe, error, reason
):
    service = make_service(fake_llm(error=error))

    result = service.triage(PAYLOAD)

    assert_failsafe(result, expected_failsafe, error=reason)


def test_provider_exception_text_is_not_part_of_the_result(
    fake_llm, make_service, expected_failsafe
):
    secret = "Incorrect API key provided: sk-proj-SECRET1234 (request req_abc)"
    service = make_service(fake_llm(error=RuntimeError(secret)))

    result = service.triage(PAYLOAD)

    assert_failsafe(result, expected_failsafe, error=PROVIDER_ERROR)
    assert "SECRET" not in str(result.error)
    assert "SECRET" not in repr(result)


def test_failure_reasons_are_the_stable_strings_that_get_stored():
    # These values are persisted in tickets.error, so renaming one is a data change.
    assert {reason.value for reason in FailureReason} == {
        "provider_error",
        "provider_timeout",
        "invalid_llm_json",
        "invalid_llm_schema",
    }


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
    # Lax validation would turn each of these into a real boolean. For model output the
    # prompt demands a boolean, so anything else is drift and must fail safe instead.
    pytest.param(
        json.dumps(valid_output(escalate="false")), SCHEMA_MISMATCH, id="string-false-escalate"
    ),
    pytest.param(
        json.dumps(valid_output(escalate="true")), SCHEMA_MISMATCH, id="string-true-escalate"
    ),
    pytest.param(
        json.dumps(valid_output(escalate="yes")), SCHEMA_MISMATCH, id="string-yes-escalate"
    ),
    pytest.param(json.dumps(valid_output(escalate=0)), SCHEMA_MISMATCH, id="zero-escalate"),
    pytest.param(json.dumps(valid_output(escalate=1)), SCHEMA_MISMATCH, id="one-escalate"),
    pytest.param(
        json.dumps(valid_output(extra_key="x")), SCHEMA_MISMATCH, id="extra-key"
    ),
    pytest.param(
        json.dumps(valid_output(reasoning="the customer is upset")),
        SCHEMA_MISMATCH,
        id="extra-key-with-prose",
    ),
    # json.loads raises RecursionError (not JSONDecodeError) for absurd nesting; it must
    # still fail safe rather than escape as an HTTP 500.
    pytest.param("[" * 100_000, INVALID_JSON, id="deeply-nested-json"),
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


def test_llm_output_schema_has_exactly_the_public_response_fields():
    # The strict model is only a stricter view of the public one: same keys, same enums.
    assert LLMTriageOutput.model_fields.keys() == TriageResponse.model_fields.keys()


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
    # Exactly the ticket content the model needs, and nothing else.
    assert json.loads(user_message["content"]) == {"text": payload.text, "channel": "chat"}


def test_client_id_never_reaches_the_provider(fake_llm, make_service):
    llm = fake_llm(content=json.dumps(valid_output()))
    service = make_service(llm)
    payload = TriageRequest(
        text="I was charged twice.", channel="form", client_id="tenant-ZZ-4711"
    )

    service.triage(payload)

    # Everything handed to the SDK call (model, messages, prompt, parameters), not just
    # the user message.
    sent = json.dumps(llm.calls[0], ensure_ascii=False)
    assert "tenant-ZZ-4711" not in sent
    assert "client_id" not in sent
    # The ticket itself still gets through.
    assert json.loads(llm.calls[0]["messages"][1]["content"]) == {
        "text": "I was charged twice.",
        "channel": "form",
    }


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
