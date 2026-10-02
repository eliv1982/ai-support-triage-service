import json
import logging
from dataclasses import dataclass
from enum import StrEnum

from openai import APITimeoutError
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.schemas import LLMTriageOutput, TriageRequest, TriageResponse
from app.services.llm_client import build_openai_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a support triage assistant.
Return only valid JSON with exactly these keys:
category, draft_reply, confidence, escalate.

Allowed category values: billing, support, complaint, other.
Allowed confidence values: high, medium, low.
draft_reply must be 1 to 6 sentences.
escalate must be a boolean.
Do not include markdown, code fences, or any extra text."""

FALLBACK_MESSAGE = (
    "Thank you for your message. This request needs review by a member of our "
    "support team before it can be answered."
)


class FailureReason(StrEnum):
    """Why the fail-safe response was used.

    These exact strings are what gets stored in tickets.error, so they are stable and carry
    nothing from the provider: exception text can quote credentials, request details and
    other implementation-specific wording. Add a value rather than renaming one.
    """

    PROVIDER_ERROR = "provider_error"
    PROVIDER_TIMEOUT = "provider_timeout"
    INVALID_LLM_JSON = "invalid_llm_json"
    INVALID_LLM_SCHEMA = "invalid_llm_schema"


class LLMOutputError(Exception):
    """The model answered, but not with a usable result."""

    def __init__(self, reason: FailureReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


@dataclass
class TriageResult:
    response: TriageResponse
    error: FailureReason | None = None
    used_fallback: bool = False


def fallback_response() -> TriageResponse:
    return TriageResponse(
        category="other",
        draft_reply=FALLBACK_MESSAGE,
        confidence="low",
        escalate=True,
    )


class TriageService:
    def __init__(self, settings: Settings | None = None, llm_client=None) -> None:
        self.settings = settings or get_settings()
        self.llm_client = llm_client or build_openai_client()

    def triage(self, payload: TriageRequest) -> TriageResult:
        try:
            raw_content = self._call_llm(payload)
        except Exception as exc:  # noqa: BLE001 - fallback path is intentional
            reason = (
                FailureReason.PROVIDER_TIMEOUT
                if isinstance(exc, (APITimeoutError, TimeoutError))
                else FailureReason.PROVIDER_ERROR
            )
            return self._fail_safe(payload, reason, exc)

        try:
            return TriageResult(response=self._parse_response(raw_content))
        except LLMOutputError as exc:
            return self._fail_safe(payload, exc.reason, exc.__cause__)

    def _fail_safe(
        self, payload: TriageRequest, reason: FailureReason, cause: BaseException | None
    ) -> TriageResult:
        # The routine log is the category, plus the exception's type and (when the SDK has
        # one) its HTTP status. Never its message or a traceback: a provider message can
        # quote a credential, and pydantic's quotes the model output, which echoes the
        # ticket text.
        logger.warning(
            "LLM triage fell back reason=%s error_type=%s status=%s client_id=%s",
            reason.value,
            type(cause).__name__ if cause is not None else None,
            getattr(cause, "status_code", None),
            payload.client_id,
        )
        return TriageResult(
            response=fallback_response(), error=reason, used_fallback=True
        )

    def _call_llm(self, payload: TriageRequest) -> str:
        response = self.llm_client.chat.completions.create(
            model=self.settings.openai_model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    # What the model needs to triage, and nothing else: client_id is only a
                    # local rate-limit key and stays out of the provider request.
                    "content": json.dumps(
                        {"text": payload.text, "channel": payload.channel},
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        return response.choices[0].message.content or ""

    def _parse_response(self, raw_content: str) -> TriageResponse:
        try:
            data = json.loads(raw_content)
        except (ValueError, TypeError, RecursionError) as exc:
            # JSONDecodeError is a ValueError; RecursionError is what absurdly nested
            # output raises instead. All of it is "not usable JSON", never a crash.
            raise LLMOutputError(FailureReason.INVALID_LLM_JSON) from exc

        try:
            output = LLMTriageOutput.model_validate(data)
        except ValidationError as exc:
            raise LLMOutputError(FailureReason.INVALID_LLM_SCHEMA) from exc

        # Strictness applies to what the model sent; callers get the plain public model.
        return TriageResponse.model_validate(output.model_dump())
