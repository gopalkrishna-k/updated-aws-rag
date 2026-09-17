"""API Key Rotator for Gemini and Groq API calls with automatic retry on 429/403 errors."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, List, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class APIKeyRotator:
    """Thread-safe Round-Robin API Key Rotator."""

    def __init__(self):
        self._lock = threading.Lock()
        self._indices = {"gemini": 0, "groq": 0}

    def get_keys(self, provider: str = "gemini") -> List[str]:
        """Fetch list of API keys for provider from environment variables."""
        provider_clean = provider.lower()
        if provider_clean in ["gemini", "google"]:
            env_val = (
                os.environ.get("GEMINI_API_KEYS")
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
                or ""
            )
        elif provider_clean == "groq":
            env_val = os.environ.get("GROQ_API_KEYS") or os.environ.get("GROQ_API_KEY") or ""
        else:
            env_val = ""

        keys = [k.strip() for k in env_val.split(",") if k.strip()]
        return keys

    def get_next_key(self, provider: str = "gemini") -> str:
        """Get the next API key in Round-Robin order."""
        provider_clean = "groq" if provider.lower() == "groq" else "gemini"
        keys = self.get_keys(provider_clean)
        if not keys:
            raise ValueError(f"No API keys configured for provider '{provider_clean}'. Check .env.")

        with self._lock:
            idx = self._indices[provider_clean] % len(keys)
            key = keys[idx]
            self._indices[provider_clean] = (idx + 1) % len(keys)
            return key

    def execute_with_retry(
        self,
        func: Callable[[str], Any],
        provider: str = "gemini",
        max_retries: Optional[int] = None,
    ) -> Any:
        """Execute a function receiving an API key, automatically retrying with next key on 429/403 errors."""
        keys = self.get_keys(provider)
        tries = max_retries if max_retries is not None else max(len(keys) * 3, 5)

        retryable_keywords = [
            "429", "resource_exhausted", "quota", "rate limit", "rate_limit",
            "403", "permission_denied", "permission denied", "401", "unauthorized",
            "invalid", "forbidden", "blocked"
        ]

        last_exception = None
        for attempt in range(tries):
            current_key = self.get_next_key(provider)
            try:
                return func(current_key)
            except Exception as exc:
                exc_str = str(exc)
                if any(err in exc_str.lower() for err in retryable_keywords):
                    logger.warning(
                        "API Key issue on attempt %d for %s. Rotating to next key... (Error: %s)",
                        attempt + 1,
                        provider,
                        exc_str[:120],
                    )
                    last_exception = exc
                    if any(q in exc_str.lower() for q in ["429", "quota", "resource_exhausted"]):
                        time.sleep(1.5)  # brief pause to allow rate-limit windows to reset
                    continue
                else:
                    raise exc

        if last_exception:
            raise last_exception
        raise RuntimeError(f"Failed to execute API call for provider '{provider}' after {tries} attempts.")


# Global rotator instance
key_rotator = APIKeyRotator()
