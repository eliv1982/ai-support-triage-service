"""Pins the real OpenAI SDK against our integration without leaving the machine.

The other tests inject a fake client, so they would not notice an SDK upgrade that
changed how `chat.completions.create(...)` is built or serialized. Here the real client
talks to a stub provider on loopback (the network guard allows loopback only).
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app.config import Settings
from app.schemas import TriageRequest
from app.services import llm_client
from app.services.triage_service import SYSTEM_PROMPT, TriageService

TRIAGE_JSON = {
    "category": "billing",
    "draft_reply": "We will look into the charge.",
    "confidence": "high",
    "escalate": False,
}

CHAT_COMPLETION = {
    "id": "chatcmpl-stub",
    "object": "chat.completion",
    "created": 1,
    "model": "contract-model",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(TRIAGE_JSON)},
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


@pytest.fixture
def stub_provider(monkeypatch: pytest.MonkeyPatch):
    """Yields (base_url, requests_received) for a loopback Chat Completions stub."""
    received: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            received.append(
                {
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                    "body": json.loads(self.rfile.read(length)),
                }
            )
            payload = json.dumps(CHAT_COMPLETION).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    # A developer's proxy settings must not divert loopback traffic.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_real_sdk_client_sends_the_expected_chat_completions_request(
    stub_provider, monkeypatch: pytest.MonkeyPatch
):
    base_url, received = stub_provider
    settings = Settings(
        _env_file=None,
        openai_api_key="stub-key-not-a-real-credential",
        openai_base_url=base_url,
        openai_model="contract-model",
    )
    # The unmodified production factory, fed by the stub's settings.
    monkeypatch.setattr(llm_client, "get_settings", lambda: settings)
    real_client = llm_client.build_openai_client()

    service = TriageService(settings=settings, llm_client=real_client)
    result = service.triage(
        TriageRequest(text="I was charged twice", channel="email", client_id="client-1")
    )

    assert result.error is None, result.error
    assert result.used_fallback is False
    assert result.response.model_dump() == TRIAGE_JSON

    assert len(received) == 1
    request = received[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["authorization"] == "Bearer stub-key-not-a-real-credential"

    body = request["body"]
    assert body["model"] == "contract-model"
    assert body["temperature"] == 0.2
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "text": "I was charged twice",
                    "channel": "email",
                    "client_id": "client-1",
                },
                ensure_ascii=False,
            ),
        },
    ]
