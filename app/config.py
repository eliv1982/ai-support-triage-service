from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The value shipped in .env.example. A copied-but-unedited .env must not start the app.
OPENAI_API_KEY_PLACEHOLDER = "your_api_key_here"


class Settings(BaseSettings):
    # BaseSettings validates defaults, so a missing key fails like a blank one.
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    # Seconds one provider call may take before the fail-safe response is used. Roughly 3x
    # what the default model needs for this short JSON task; raise it for slower models.
    openai_timeout_seconds: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    rate_limit_per_minute: int = 5
    database_url: str = "sqlite:///./app.db"

    # hide_input_in_errors keeps rejected values (possibly a secret) out of error text.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", hide_input_in_errors=True
    )

    @field_validator("openai_api_key")
    @classmethod
    def _require_real_api_key(cls, value: str) -> str:
        key = value.strip()
        if not key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Set it in .env or the environment."
            )
        if key == OPENAI_API_KEY_PLACEHOLDER:
            raise ValueError(
                "OPENAI_API_KEY is still the placeholder from .env.example. "
                "Replace it with a real key."
            )
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
