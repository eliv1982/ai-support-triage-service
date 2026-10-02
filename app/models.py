from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, Text, false, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    client_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    channel: Mapped[str] = mapped_column(String(20), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[str] = mapped_column(String(20), nullable=False)
    escalate: Mapped[bool] = mapped_column(Boolean, nullable=False)
    draft_reply: Mapped[str] = mapped_column(Text, nullable=False)
    # A stable reason code (triage_service.FailureReason) saying why the fail-safe was
    # used; never provider text. NULL for a normal result.
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # True when the stored result is the fail-safe rather than the model's answer. The
    # content cannot tell them apart: a model may answer exactly like the fail-safe.
    # Kept as the last column so a database upgraded in place (init_db) has the same
    # column order as a fresh one.
    used_fallback: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=false()
    )
