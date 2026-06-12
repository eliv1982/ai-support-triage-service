from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from threading import Lock


class InMemoryRateLimiter:
    def __init__(self, limit_per_minute: int) -> None:
        self.limit_per_minute = limit_per_minute
        self._requests: dict[str, deque[datetime]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, client_id: str) -> bool:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=1)

        with self._lock:
            client_requests = self._requests[client_id]
            while client_requests and client_requests[0] < cutoff:
                client_requests.popleft()

            if len(client_requests) >= self.limit_per_minute:
                return False

            client_requests.append(now)
            return True

    def reset(self) -> None:
        with self._lock:
            self._requests.clear()
