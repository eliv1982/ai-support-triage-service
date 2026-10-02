import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from sqlalchemy.orm import Session

from app.database import engine, get_db, init_db
from app.dependencies import get_rate_limiter, get_triage_service
from app.logging_config import setup_logging
from app.repository import save_ticket
from app.schemas import HealthResponse, TriageRequest, TriageResponse
from app.services.triage_service import TriageService

setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db(engine)
    yield


app = FastAPI(title="ai-support-triage-service", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/triage", response_model=TriageResponse)
def triage(
    payload: TriageRequest,
    db: Session = Depends(get_db),
    rate_limiter=Depends(get_rate_limiter),
    triage_service: TriageService = Depends(get_triage_service),
) -> TriageResponse:
    logger.info(
        "Incoming triage request client_id=%s channel=%s text_length=%s",
        payload.client_id,
        payload.channel,
        len(payload.text),
    )

    if not rate_limiter.allow(payload.client_id):
        # Rejected before triage, so it is not a ticket: nothing is stored and the provider
        # is never called. (client_id is validated, so it cannot carry a newline into the log.)
        logger.warning("Rate limit exceeded for client_id=%s", payload.client_id)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
        )

    result = triage_service.triage(payload)
    save_ticket(
        db=db,
        payload=payload,
        result=result.response,
        error=result.error,
        used_fallback=result.used_fallback,
    )
    return result.response
