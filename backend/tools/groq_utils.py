import os
import time
from typing import Callable, TypeVar

T = TypeVar("T")

GROQ_MISSING_MSG = "AI service is not configured (GROQ_API_KEY missing)."


def groq_configured() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def require_groq() -> None:
    if not groq_configured():
        raise RuntimeError(GROQ_MISSING_MSG)


def is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "rate_limit" in msg


def groq_call_with_retry(fn: Callable[[], T], max_attempts: int = 3) -> T:
    """Sync retry wrapper for Groq API calls."""
    require_groq()
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if is_rate_limit_error(e) and attempt < max_attempts - 1:
                time.sleep(3 * (attempt + 1))
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("Groq call failed")
