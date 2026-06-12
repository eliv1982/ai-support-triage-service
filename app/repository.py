from sqlalchemy.orm import Session

from app.models import Ticket
from app.schemas import TriageRequest, TriageResponse


def save_ticket(
    db: Session,
    payload: TriageRequest,
    result: TriageResponse,
    error: str | None = None,
) -> Ticket:
    ticket = Ticket(
        client_id=payload.client_id,
        channel=payload.channel,
        text=payload.text,
        category=result.category,
        confidence=result.confidence,
        escalate=result.escalate,
        draft_reply=result.draft_reply,
        error=error,
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    return ticket
