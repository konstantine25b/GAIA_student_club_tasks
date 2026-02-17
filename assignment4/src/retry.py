import random
import time
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 5
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    jitter_s: float = 0.25


class RetryableError(Exception):
    """
    Wrap errors we consider transient (rate limits, 5xx, timeouts, etc.).
    """


def retry_call(fn: Callable[[], T], *, config: RetryConfig, on_retry: Optional[Callable[[int, Exception], None]] = None) -> T:
    """
    Calls `fn` with exponential backoff and jitter.
    """
    if config.max_attempts <= 0:
        raise ValueError("max_attempts must be > 0")

    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except RetryableError as e:
            if attempt >= config.max_attempts:
                raise

            delay = min(config.max_delay_s, config.base_delay_s * (2 ** (attempt - 1)))
            delay += random.uniform(0.0, config.jitter_s)
            if on_retry:
                on_retry(attempt, e)
            time.sleep(delay)

