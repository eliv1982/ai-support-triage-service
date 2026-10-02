from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator


Channel = Literal["email", "form", "chat"]
Category = Literal["billing", "support", "complaint", "other"]
Confidence = Literal["high", "medium", "low"]

# Common identifiers fit well inside this (UUID 36, ULID 26, most e-mail addresses < 100),
# and it is comfortably below the 255-character database column.
CLIENT_ID_MAX_LENGTH = 128

# Surrounding whitespace is trimmed before the length checks: a blank ticket is empty
# after trimming and so rejected, and padding does not count towards the maximum.
TicketText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
]


class TriageRequest(BaseModel):
    text: TicketText
    channel: Channel
    client_id: str = Field(..., min_length=1, max_length=CLIENT_ID_MAX_LENGTH)

    @field_validator("client_id")
    @classmethod
    def client_id_has_no_whitespace_or_control_characters(cls, value: str) -> str:
        # client_id ends up in log lines and is only ever a self-declared label, so keep
        # it to one visible token. isprintable() is False for control characters (\n, \r,
        # ESC, NUL), invisible format characters (zero-width, bidi overrides) and every
        # whitespace character except the ASCII space, which is excluded explicitly.
        # Everything else, including non-ASCII letters, is accepted.
        if " " in value or not value.isprintable():
            raise ValueError("client_id must not contain whitespace or control characters")
        return value


class TriageResponse(BaseModel):
    category: Category
    draft_reply: str = Field(..., min_length=1)
    confidence: Confidence
    escalate: bool


class LLMTriageOutput(TriageResponse):
    """The model's answer: the response fields, validated strictly.

    Model output is untrusted and the prompt asks for exactly these keys with a real
    boolean, so anything else is drift and falls back to the fail-safe response. That
    means no coercion (`"false"`, `"yes"`, `0` and `1` are not booleans) and no extra
    keys. The public response model above is deliberately left as it was.
    """

    model_config = ConfigDict(strict=True, extra="forbid")


class HealthResponse(BaseModel):
    status: str
