from functools import lru_cache

from app.config import get_settings
from app.rate_limiter import InMemoryRateLimiter
from app.services.triage_service import TriageService


@lru_cache
def get_rate_limiter() -> InMemoryRateLimiter:
    settings = get_settings()
    return InMemoryRateLimiter(limit_per_minute=settings.rate_limit_per_minute)


@lru_cache
def get_triage_service() -> TriageService:
    # One service, hence one SDK client and connection pool, for the whole process.
    # TriageService keeps no per-request state, and the SDK client is thread-safe.
    return TriageService()
