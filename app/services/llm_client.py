from openai import OpenAI

from app.config import get_settings


def build_openai_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
