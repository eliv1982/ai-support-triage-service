from functools import lru_cache

from app.config import get_settings
from app.rate_limiter import InMemoryRateLimiter
from app.services.triage_service import TriageService


@lru_cache
def get_rate_limiter() -> InMemoryRateLimiter:
    settings = get_settings()
    return InMemoryRateLimiter(limit_per_minute=settings.rate_limit_per_minute)


def get_triage_service() -> TriageService:
    return TriageService()
