"""Thread safety tests for Membrane rate limiting."""
import threading
import time
from operon_ai import Membrane, Signal, ThreatLevel


def test_rate_limit_thread_safety():
    """Concurrent requests should not cause race conditions."""
    membrane = Membrane(
        threshold=ThreatLevel.SUSPICIOUS,
        rate_limit=50,
        silent=True,
    )

    results = []
    errors = []
    lock = threading.Lock()

    def make_requests(n: int):
        try:
            for _ in range(n):
                result = membrane.filter(Signal(content="test"))
                with lock:
                    results.append(result)
        except Exception as e:
            with lock:
                errors.append(e)

    # Launch 20 threads each making 10 requests = 200 total
    # With rate_limit=50, we expect exactly 50 allowed and 150 blocked
    threads = [threading.Thread(target=make_requests, args=(10,)) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Thread errors: {errors}"
    assert len(results) == 200
    # First 50 should be allowed, rest should be rate-limited
    allowed = sum(1 for r in results if r.allowed)
    rate_limited = sum(1 for r in results if not r.allowed)
    # With thread safety, we should have exactly 50 allowed
    # Without thread safety, we might have more due to race conditions
    print(f"Allowed: {allowed}, Rate-limited: {rate_limited}")
    assert rate_limited == 150, f"Expected 150 rate-limited, got {rate_limited}"
    assert allowed == 50, f"Expected 50 allowed, got {allowed}"


def test_rate_limit_exact_boundary():
    """Rate limit should trigger at exactly the threshold."""
    membrane = Membrane(
        threshold=ThreatLevel.SUSPICIOUS,
        rate_limit=5,
        silent=True,
    )

    # First 5 should pass
    for i in range(5):
        result = membrane.filter(Signal(content=f"request {i}"))
        assert result.allowed, f"Request {i} should be allowed"

    # 6th should be rate-limited
    result = membrane.filter(Signal(content="request 5"))
    assert not result.allowed
    # The rate limit result doesn't have a block_reason attribute, check threat_level
    assert result.threat_level == ThreatLevel.CRITICAL
