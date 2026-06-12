import json
import logging
from dataclasses import dataclass

from pydantic import ValidationError

from app.config import Settings, get_settings
from app.schemas import TriageRequest, TriageResponse
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
    "Спасибо за обращение. Мы передали его оператору для ручной проверки "
    "и вернемся с ответом как можно скорее."
)


@dataclass
class TriageResult:
    response: TriageResponse
    error: str | None = None
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
            parsed = self._parse_response(raw_content)
            return TriageResult(response=parsed)
        except Exception as exc:  # noqa: BLE001 - fallback path is intentional
            logger.exception("LLM triage failed for client_id=%s", payload.client_id)
            fallback = fallback_response()
            logger.warning("Fallback response used for client_id=%s", payload.client_id)
            return TriageResult(
                response=fallback,
                error=str(exc),
                used_fallback=True,
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
                    "content": json.dumps(
                        {
                            "text": payload.text,
                            "channel": payload.channel,
                            "client_id": payload.client_id,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        return response.choices[0].message.content or ""

    def _parse_response(self, raw_content: str) -> TriageResponse:
        try:
            data = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON returned by LLM") from exc

        try:
            return TriageResponse.model_validate(data)
        except ValidationError as exc:
            raise ValueError("LLM response did not match schema") from exc
