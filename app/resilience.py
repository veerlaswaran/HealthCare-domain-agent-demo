"""
Task 16 — Retry and timeout utilities.

retry_with_backoff
    Exponential-backoff retry decorator.
    Parameters (all configurable, defaults stated below):
      max_attempts    = 3      — total call attempts (1 real + 2 retries)
      initial_delay   = 0.1 s  — wait after first failure
      max_delay       = 2.0 s  — cap on any single wait
      backoff_factor  = 2.0    — multiply delay by this after each failure
      jitter          = True   — add uniform random fraction of delay to
                                 prevent thundering-herd on concurrent retries

    Retry is attempted on any Exception subclass (configurable via
    retry_on parameter).  On exhaustion raises the last exception wrapped
    in RetryExhaustedError.

node_timeout
    Context manager that raises NodeTimeoutError if the wrapped block
    takes longer than `seconds`.  Uses threading.Timer so it works on
    Python 3.9 (signal.alarm is UNIX-only and not usable inside threads).

GlobalTimeoutError / run_with_global_timeout
    Runs a callable with an overall wall-clock timeout.  Raises
    GlobalTimeoutError if the callable does not return in time.
"""

from __future__ import annotations

import random
import threading
import time
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""
    def __init__(self, attempts: int, last_exc: Exception) -> None:
        self.attempts = attempts
        self.last_exc = last_exc
        super().__init__(
            f"All {attempts} attempt(s) failed. "
            f"Last error: {type(last_exc).__name__}: {last_exc}"
        )


class NodeTimeoutError(Exception):
    """Raised when a single node exceeds its per-node timeout."""
    def __init__(self, node_name: str, seconds: float) -> None:
        self.node_name = node_name
        self.seconds = seconds
        super().__init__(f"Node '{node_name}' timed out after {seconds}s")


class GlobalTimeoutError(Exception):
    """Raised when the whole graph run exceeds its global timeout."""
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        super().__init__(f"Global timeout exceeded ({seconds}s)")


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 0.1,
    max_delay: float = 2.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retry_on: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """
    Exponential-backoff retry decorator.

    Config:
      max_attempts   = 3     (1 try + 2 retries)
      initial_delay  = 0.1 s
      max_delay      = 2.0 s
      backoff_factor = 2.0
      jitter         = True  (uniform [0, delay) added to each wait)
      retry_on       = (Exception,) — retry on any exception

    Usage:
        @retry_with_backoff(max_attempts=3, initial_delay=0.1)
        def flaky_call():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            last_exc: Optional[Exception] = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retry_on as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        break
                    actual_delay = min(delay, max_delay)
                    if jitter:
                        actual_delay += random.uniform(0, actual_delay * 0.1)
                    print(
                        f"  [RETRY] {func.__name__} attempt {attempt}/{max_attempts} "
                        f"failed: {exc}. Retrying in {actual_delay:.3f}s..."
                    )
                    time.sleep(actual_delay)
                    delay = min(delay * backoff_factor, max_delay)

            raise RetryExhaustedError(max_attempts, last_exc)  # type: ignore[arg-type]

        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Per-node timeout context manager
# ---------------------------------------------------------------------------

class node_timeout:
    """
    Context manager that raises NodeTimeoutError if the block runs longer
    than `seconds`.

    Uses a background threading.Timer — safe in any thread, no UNIX signal
    required (unlike signal.alarm).

    Usage:
        with node_timeout("rag_tool", seconds=2.0):
            result = slow_rag_call()
    """

    def __init__(self, node_name: str, seconds: float) -> None:
        self.node_name = node_name
        self.seconds = seconds
        self._timer: Optional[threading.Timer] = None
        self._timed_out = threading.Event()
        self._exc: Optional[NodeTimeoutError] = None

    def _on_timeout(self) -> None:
        self._exc = NodeTimeoutError(self.node_name, self.seconds)
        self._timed_out.set()

    def __enter__(self) -> "node_timeout":
        self._timer = threading.Timer(self.seconds, self._on_timeout)
        self._timer.daemon = True
        self._timer.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._timer:
            self._timer.cancel()
        if self._timed_out.is_set() and self._exc:
            raise self._exc
        return False

    def check(self) -> None:
        """Call this inside long-running loops to raise early on timeout."""
        if self._timed_out.is_set() and self._exc:
            raise self._exc


# ---------------------------------------------------------------------------
# Global timeout wrapper
# ---------------------------------------------------------------------------

def run_with_global_timeout(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    timeout_seconds: float = 10.0,
) -> Any:
    """
    Run `func(*args, **kwargs)` with a global wall-clock timeout.

    Raises GlobalTimeoutError if the function does not complete within
    `timeout_seconds`.

    Implementation: runs func in a daemon thread; the main thread waits
    up to timeout_seconds.  If the thread is still alive, we raise.
    The thread itself is allowed to finish naturally (daemon=True means it
    won't block process exit).
    """
    if kwargs is None:
        kwargs = {}

    result_box: list[Any] = [None]
    exc_box: list[Optional[Exception]] = [None]

    def _run() -> None:
        try:
            result_box[0] = func(*args, **kwargs)
        except Exception as e:
            exc_box[0] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_seconds)

    if t.is_alive():
        raise GlobalTimeoutError(timeout_seconds)

    if exc_box[0] is not None:
        raise exc_box[0]

    return result_box[0]
