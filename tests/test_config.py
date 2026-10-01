import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import OPENAI_API_KEY_PLACEHOLDER, Settings, get_settings

REPO_ROOT = Path(__file__).resolve().parent.parent
DUMMY_KEY = "test-key-not-a-real-credential"


@pytest.fixture(autouse=True)
def no_ambient_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """conftest sets a fake key globally; each test here decides for itself."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_missing_key_is_rejected():
    with pytest.raises(ValidationError, match="OPENAI_API_KEY is not set"):
        Settings(_env_file=None)


@pytest.mark.parametrize("blank", ["", " ", "   ", "\t", " \n "])
def test_blank_key_is_rejected(monkeypatch: pytest.MonkeyPatch, blank: str):
    monkeypatch.setenv("OPENAI_API_KEY", blank)
    with pytest.raises(ValidationError, match="OPENAI_API_KEY is not set"):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    "placeholder",
    [OPENAI_API_KEY_PLACEHOLDER, f"  {OPENAI_API_KEY_PLACEHOLDER}\n"],
)
def test_documented_placeholder_is_rejected(
    monkeypatch: pytest.MonkeyPatch, placeholder: str
):
    monkeypatch.setenv("OPENAI_API_KEY", placeholder)
    with pytest.raises(ValidationError, match="still the placeholder"):
        Settings(_env_file=None)


def test_dummy_non_secret_key_is_accepted_unchanged(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY_KEY)
    assert Settings(_env_file=None).openai_api_key == DUMMY_KEY


def test_key_format_is_not_policed(monkeypatch: pytest.MonkeyPatch):
    # Providers change key formats and OpenAI-compatible servers accept anything;
    # only absence and the documented placeholder are rejected.
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    assert Settings(_env_file=None).openai_api_key == "x"


def test_shipped_env_example_is_not_a_startable_config():
    # The file users copy to .env must fail until they put a real key in it.
    with pytest.raises(ValidationError, match="still the placeholder"):
        Settings(_env_file=REPO_ROOT / ".env.example")


def test_key_is_read_from_a_dotenv_file(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"OPENAI_API_KEY={DUMMY_KEY}\n", encoding="utf-8")
    assert Settings(_env_file=env_file).openai_api_key == DUMMY_KEY

    env_file.write_text(f"OPENAI_API_KEY={OPENAI_API_KEY_PLACEHOLDER}\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="still the placeholder"):
        Settings(_env_file=env_file)


def test_validation_errors_do_not_echo_input_values():
    secret_looking = "pretend-secret-value-for-test"
    with pytest.raises(ValidationError) as excinfo:
        Settings(
            _env_file=None,
            openai_api_key=secret_looking,
            rate_limit_per_minute="not-a-number",
        )
    message = str(excinfo.value)
    assert "rate_limit_per_minute" in message
    assert "not-a-number" not in message
    assert secret_looking not in message


def test_get_settings_fails_without_a_key_and_does_not_cache_the_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(REPO_ROOT / "tests")  # no .env here to supply a key
    get_settings.cache_clear()
    try:
        with pytest.raises(ValidationError, match="OPENAI_API_KEY is not set"):
            get_settings()
        monkeypatch.setenv("OPENAI_API_KEY", DUMMY_KEY)
        assert get_settings().openai_api_key == DUMMY_KEY
    finally:
        get_settings.cache_clear()


def _import_app_in_fresh_process(tmp_path: Path, key: str | None):
    """Run a bare `import app.main` (what uvicorn does at startup) outside the repo,
    so a developer's local .env cannot supply a key."""
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    env["PYTHONPATH"] = str(REPO_ROOT)
    if key is not None:
        env["OPENAI_API_KEY"] = key
    return subprocess.run(
        [sys.executable, "-c", "import app.main"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_application_startup_fails_without_a_key(tmp_path: Path):
    result = _import_app_in_fresh_process(tmp_path, key=None)
    assert result.returncode != 0
    assert "OPENAI_API_KEY is not set" in result.stderr


def test_application_startup_fails_with_the_placeholder(tmp_path: Path):
    result = _import_app_in_fresh_process(tmp_path, key=OPENAI_API_KEY_PLACEHOLDER)
    assert result.returncode != 0
    assert "still the placeholder" in result.stderr


def test_application_startup_succeeds_with_a_dummy_key(tmp_path: Path):
    # Negative control for the two tests above: the same process starts fine with a key,
    # so those failures are caused by the key and not by the harness.
    result = _import_app_in_fresh_process(tmp_path, key=DUMMY_KEY)
    assert result.returncode == 0, result.stderr
