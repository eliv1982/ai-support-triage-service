import time
from collections import OrderedDict, deque
from collections.abc import Callable
from threading import Lock

WINDOW_SECONDS = 60.0


class InMemoryRateLimiter:
    """Sliding window: at most `limit_per_minute` accepted requests per client in any 60 s.

    State lives in this process only. Memory is bounded without a background task: every
    call first drops the clients whose newest accepted request has already left the window
    (they can no longer influence any decision), so only clients active within the last
    minute are held, however many distinct ids have come and gone.
    """

    def __init__(
        self,
        limit_per_minute: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limit_per_minute = limit_per_minute
        self._clock = clock
        # Accepted-request times per client, ordered by each client's newest accepted
        # request, oldest first. Only an accepted request moves a client to the end (a
        # rejection changes nothing), so this order is the order of those times and
        # cleanup can stop at the first client that is still active.
        self._requests: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = Lock()

    def allow(self, client_id: str) -> bool:
        # The clock is read under the lock so recorded times follow the order in which
        # requests were decided, which the ordering above relies on.
        with self._lock:
            now = self._clock()
            cutoff = now - WINDOW_SECONDS
            self._drop_idle_clients(cutoff)

            history = self._requests.get(client_id)
            if history is None:
                if self.limit_per_minute <= 0:
                    return False
                history = self._requests[client_id] = deque()
            else:
                while history and history[0] < cutoff:
                    history.popleft()
                if len(history) >= self.limit_per_minute:
                    return False

            history.append(now)
            self._requests.move_to_end(client_id)
            return True

    def _drop_idle_clients(self, cutoff: float) -> None:
        while self._requests:
            client_id, history = next(iter(self._requests.items()))
            if history and history[-1] >= cutoff:
                return
            del self._requests[client_id]

    def reset(self) -> None:
        with self._lock:
            self._requests.clear()
