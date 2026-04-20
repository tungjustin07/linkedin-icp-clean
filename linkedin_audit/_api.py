"""
Shared Anthropic API client and retry logic for the LinkedIn audit pipeline.

Mirrors the lazy-singleton pattern from claude_analyze.py and adds
exponential backoff for rate limits and transient errors.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("linkedin_audit._api")

HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-6"
MAX_RETRIES = 5

_client: Optional[anthropic.Anthropic] = None


def get_client() -> anthropic.Anthropic:
    """Lazy singleton Anthropic client."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


def call_with_backoff(
    client: anthropic.Anthropic,
    model: str,
    system: "str | list[dict]",
    messages: list[dict],
    tools: list[dict],
    max_tokens: int = 2000,
    retries: int = MAX_RETRIES,
) -> anthropic.types.Message:
    """
    Call client.messages.create() with exponential backoff.

    `system` may be a plain string (legacy) or a list of content blocks — the
    latter is required when using prompt caching via `cache_control` markers.
    The Anthropic SDK accepts both forms natively; no special handling here.

    Retryable: RateLimitError, APIStatusError (529), APIConnectionError.
    Non-retryable: AuthenticationError, PermissionDeniedError.
    Wait formula: min(2^attempt * 5, 120) seconds.
    """
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(retries):
        try:
            return client.messages.create(
                model=model,
                system=system,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
            )
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError):
            raise  # never retry auth errors
        except anthropic.APIStatusError as e:
            if e.status_code not in (429, 529):
                raise
            last_exc = e
        except (anthropic.RateLimitError, anthropic.APIConnectionError) as e:
            last_exc = e

        wait = min((2 ** attempt) * 5, 120)
        log.warning(
            "API error (%s) on attempt %d/%d — sleeping %ds: %s",
            type(last_exc).__name__,
            attempt + 1,
            retries,
            wait,
            last_exc,
        )
        time.sleep(wait)

    raise last_exc
