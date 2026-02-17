import random
import time
from collections import deque
from dataclasses import dataclass


@dataclass
class RateLimitConfig:
    """
    Simple client-side rate limiter.

    This is NOT token-aware; it limits requests per time window.
    """

    max_requests: int = 10
    per_seconds: float = 60.0


class SlidingWindowRateLimiter:
    def __init__(self, config: RateLimitConfig):
        if config.max_requests <= 0:
            raise ValueError("max_requests must be > 0")
        if config.per_seconds <= 0:
            raise ValueError("per_seconds must be > 0")
        self._config = config
        self._timestamps = deque()

    def acquire(self) -> None:
        """
        Blocks until a request is allowed.
        """
        while True:
            now = time.time()
            window_start = now - self._config.per_seconds

            while self._timestamps and self._timestamps[0] < window_start:
                self._timestamps.popleft()

            if len(self._timestamps) < self._config.max_requests:
                self._timestamps.append(now)
                return

            # Sleep until the earliest timestamp exits the window, plus small jitter.
            sleep_for = (self._timestamps[0] + self._config.per_seconds) - now
            sleep_for = max(0.05, sleep_for) + random.uniform(0.0, 0.2)
            time.sleep(sleep_for)

