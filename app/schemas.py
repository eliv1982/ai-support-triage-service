from typing import Literal

from pydantic import BaseModel, Field


Channel = Literal["email", "form", "chat"]
Category = Literal["billing", "support", "complaint", "other"]
Confidence = Literal["high", "medium", "low"]


class TriageRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    channel: Channel
    client_id: str = Field(..., min_length=1)


class TriageResponse(BaseModel):
    category: Category
    draft_reply: str = Field(..., min_length=1)
    confidence: Confidence
    escalate: bool


class HealthResponse(BaseModel):
    status: str
