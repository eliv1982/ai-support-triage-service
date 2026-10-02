"""The in-memory limiter: window semantics, idle-key cleanup, and thread safety.

Time is simulated with an injected clock, so nothing here sleeps or depends on wall-clock
timing.
"""

import threading

import pytest

from app.rate_limiter import InMemoryRateLimiter

WINDOW = 60  # seconds; the "per minute" in RATE_LIMIT_PER_MINUTE


class FakeClock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def make_limiter(clock: FakeClock, limit: int = 3) -> InMemoryRateLimiter:
    return InMemoryRateLimiter(limit_per_minute=limit, clock=clock)


def tracked(limiter: InMemoryRateLimiter) -> set[str]:
    """The client ids the limiter currently holds state for (its memory footprint)."""
    return set(limiter._requests)


# --- A. window semantics are unchanged ------------------------------------------------


def test_allows_up_to_the_limit_then_rejects(clock):
    limiter = make_limiter(clock, limit=3)

    assert [limiter.allow("a") for _ in range(5)] == [True, True, True, False, False]


def test_each_client_has_its_own_quota(clock):
    limiter = make_limiter(clock, limit=1)

    assert limiter.allow("a") is True
    assert limiter.allow("a") is False
    assert limiter.allow("b") is True


def test_a_request_stops_counting_once_it_is_older_than_the_window(clock):
    limiter = make_limiter(clock, limit=2)
    assert limiter.allow("a") and limiter.allow("a")

    clock.advance(WINDOW - 1)
    assert limiter.allow("a") is False  # both still inside the window

    clock.advance(2)  # now 61 s after the first two
    assert limiter.allow("a") is True


def test_the_window_slides_rather_than_resetting_all_at_once(clock):
    limiter = make_limiter(clock, limit=2)
    assert limiter.allow("a")  # t = 0
    clock.advance(30)
    assert limiter.allow("a")  # t = 30

    clock.advance(31)  # t = 61: the first has expired, the second has not
    assert limiter.allow("a") is True  # one slot freed...
    assert limiter.allow("a") is False  # ...and only one


def test_rejected_attempts_do_not_extend_the_window(clock):
    limiter = make_limiter(clock, limit=1)
    assert limiter.allow("a")

    for _ in range(5):  # hammering while limited must not keep the client locked out
        clock.advance(10)
        assert limiter.allow("a") is False

    clock.advance(11)  # 61 s after the one request that counted
    assert limiter.allow("a") is True


def test_a_zero_limit_rejects_everything_and_remembers_nothing(clock):
    limiter = make_limiter(clock, limit=0)

    assert [limiter.allow(f"c{i}") for i in range(5)] == [False] * 5
    assert tracked(limiter) == set()


def test_reset_forgets_everything(clock):
    limiter = make_limiter(clock, limit=1)
    assert limiter.allow("a")
    assert limiter.allow("a") is False

    limiter.reset()

    assert tracked(limiter) == set()
    assert limiter.allow("a") is True


# --- B. idle clients are forgotten ----------------------------------------------------


def test_clients_that_went_quiet_are_eventually_removed(clock):
    limiter = make_limiter(clock)
    for i in range(100):
        limiter.allow(f"client-{i}")
    assert len(tracked(limiter)) == 100

    clock.advance(WINDOW + 1)
    limiter.allow("newcomer")  # any later access triggers the cleanup

    assert tracked(limiter) == {"newcomer"}


def test_cleanup_keeps_clients_whose_requests_still_count(clock):
    limiter = make_limiter(clock, limit=1)
    assert limiter.allow("old")  # t = 0
    clock.advance(30)
    assert limiter.allow("recent")  # t = 30

    clock.advance(31)  # t = 61: "old" has expired, "recent" has not
    assert limiter.allow("newcomer") is True

    assert tracked(limiter) == {"recent", "newcomer"}
    assert limiter.allow("recent") is False  # its request still counts after the cleanup
    assert limiter.allow("old") is True  # expired history no longer blocks


def test_a_rejected_attempt_does_not_hide_an_expired_client_from_cleanup(clock):
    """Cleanup walks clients in order of their last accepted request. A rejection is not
    an accepted request, so it must not reorder that, or an expired client could sit
    behind a live one and never be reclaimed."""
    limiter = make_limiter(clock, limit=1)
    assert limiter.allow("a")  # t = 0
    clock.advance(10)
    assert limiter.allow("b")  # t = 10
    clock.advance(10)
    assert limiter.allow("a") is False  # t = 20: rejected, still inside a's window

    clock.advance(45)  # t = 65: a (t=0) expired, b (t=10) has not
    assert limiter.allow("c") is True

    assert tracked(limiter) == {"b", "c"}


def test_memory_stays_bounded_over_a_long_run_of_unique_clients(clock):
    limiter = make_limiter(clock, limit=5)
    peak = 0

    # One request per second from a never-seen-before client, for almost three hours.
    for i in range(10_000):
        limiter.allow(f"client-{i}")
        peak = max(peak, len(tracked(limiter)))
        clock.advance(1)

    # Only clients active within the last window can be held: about WINDOW of them,
    # never the 10,000 that have passed through.
    assert peak <= WINDOW + 1
    assert len(tracked(limiter)) <= WINDOW + 1


def test_a_long_running_client_never_accumulates_more_than_its_limit(clock):
    limiter = make_limiter(clock, limit=3)

    for _ in range(1_000):
        limiter.allow("busy")
        clock.advance(1)

    assert len(limiter._requests["busy"]) <= 3


# --- C. concurrency -------------------------------------------------------------------


def run_in_threads(count: int, target) -> list[BaseException]:
    errors: list[BaseException] = []
    barrier = threading.Barrier(count)

    def runner(index: int) -> None:
        try:
            barrier.wait(timeout=10)
            target(index)
        except BaseException as exc:  # noqa: BLE001 - surfaced to the asserting test
            errors.append(exc)

    threads = [threading.Thread(target=runner, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    return errors


def test_concurrent_requests_for_one_client_never_exceed_the_limit(clock):
    limiter = make_limiter(clock, limit=5)
    allowed = []
    lock = threading.Lock()

    def hammer(_: int) -> None:
        for _ in range(50):
            if limiter.allow("shared"):
                with lock:
                    allowed.append(1)

    errors = run_in_threads(16, hammer)

    assert errors == []
    assert len(allowed) == 5  # 800 attempts, exactly the limit got through


def test_cleanup_running_alongside_new_clients_stays_consistent(clock):
    limiter = make_limiter(clock, limit=2)
    stop = threading.Event()

    def advance_time() -> None:
        while not stop.is_set():
            clock.advance(5)  # the window keeps expiring under the workers' feet

    def churn(index: int) -> None:
        for i in range(500):
            limiter.allow(f"worker-{index}-{i}")  # new clients: constant cleanup work
            limiter.allow("contended")  # one key everybody fights over

    ticker = threading.Thread(target=advance_time)
    ticker.start()
    try:
        errors = run_in_threads(8, churn)
    finally:
        stop.set()
        ticker.join(timeout=10)

    assert errors == []
    clock.advance(WINDOW + 1)
    assert limiter.allow("after") is True
    assert tracked(limiter) == {"after"}  # everything else expired and was reclaimed
