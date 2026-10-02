from openai import OpenAI

from app.config import get_settings

# No SDK-level retries. Each triage call occupies one of the app's ~40 sync worker threads,
# and the SDK both retries timeouts and sleeps for a server-sent Retry-After (up to 60 s),
# so retries would make the worst-case wait unbounded by us. A failed call already degrades
# to the escalating fail-safe response, so a transient error costs one manual review.
OPENAI_MAX_RETRIES = 0


def build_openai_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=settings.openai_timeout_seconds,
        max_retries=OPENAI_MAX_RETRIES,
    )
