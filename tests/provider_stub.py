"""A Chat Completions provider on loopback, for tests that need the real SDK transport.

Controls: `status` (200 = success, anything else = that error), `delay` seconds before
answering (released early by `release()` so teardown never waits it out), and
`retry_after` (sent as a Retry-After header). Every request that reaches the server is
counted in `attempts`, which is how the retry policy is observed from the outside.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
    "model": "stub-model",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(TRIAGE_JSON)},
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


class ProviderStub:
    def __init__(self, status: int = 200, delay: float = 0.0, retry_after=None):
        self.status = status
        self.delay = delay
        self.retry_after = retry_after
        self._attempts = 0
        self._lock = threading.Lock()
        self._released = threading.Event()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                with stub._lock:
                    stub._attempts += 1
                if stub.delay:
                    stub._released.wait(stub.delay)
                if stub.status == 200:
                    code, payload = 200, CHAT_COMPLETION
                else:
                    code, payload = stub.status, {"error": {"message": "stub error"}}
                body = json.dumps(payload).encode()
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    if stub.retry_after is not None:
                        self.send_header("Retry-After", str(stub.retry_after))
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except OSError:
                    pass  # the client gave up (timeout): expected in the stall tests

            def log_message(self, *args):
                pass

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):
                pass  # resets from clients that timed out are not interesting

        self._server = Server(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    @property
    def attempts(self) -> int:
        with self._lock:
            return self._attempts

    def release(self) -> None:
        """Let every handler that is waiting out `delay` answer now."""
        self._released.set()

    def close(self) -> None:
        self.release()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
