from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from google import genai
from google.genai import types

from .rate_limit import RateLimitConfig, SlidingWindowRateLimiter
from .retry import RetryConfig, RetryableError, retry_call


@dataclass(frozen=True)
class GenerationParams:
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    max_output_tokens: int = 1024


class GeminiLLM:
    """
    Minimal Gemini text-generation wrapper with:
      - client-side rate limiting (requests/window)
      - retry/backoff for transient errors
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        rate_limit: RateLimitConfig | None = None,
        retry: RetryConfig | None = None,
    ):
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._limiter = SlidingWindowRateLimiter(rate_limit or RateLimitConfig())
        self._retry_cfg = retry or RetryConfig()

    def generate_text(
        self,
        *,
        prompt: str,
        params: GenerationParams,
        extra_config: dict[str, Any] | None = None,
    ) -> str:
        resp = self.generate_response(
            contents=prompt,
            params=params,
            extra_config=extra_config,
        )
        text = _extract_text(resp)
        if not text.strip():
            raise RetryableError("Empty response text from model (treating as transient).")
        return text

    def generate_response(
        self,
        *,
        contents: str | list[types.Content],
        params: GenerationParams,
        system_instruction: str | None = None,
        extra_config: dict[str, Any] | None = None,
    ) -> Any:
        self._limiter.acquire()

        def _call() -> Any:
            try:
                cfg = types.GenerateContentConfig(
                    temperature=params.temperature,
                    top_p=params.top_p,
                    top_k=params.top_k,
                    max_output_tokens=params.max_output_tokens,
                    system_instruction=system_instruction,
                    **(extra_config or {}),
                )

                return self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=cfg,
                )
            except Exception as e:  # library raises various exceptions; classify conservatively
                if _looks_permanent_quota_or_billing_issue(e):
                    raise RuntimeError(
                        "Gemini quota/billing issue (not retrying).\n"
                        "- Open https://ai.dev/rate-limit to confirm you have non-zero quota\n"
                        "- If free-tier quota is 0, enable billing / use a project with Gemini API enabled\n"
                        f"- Underlying error: {_summarize_error(e)}"
                    ) from e
                if _looks_retryable(e):
                    raise RetryableError(_summarize_error(e)) from e
                raise

        return retry_call(_call, config=self._retry_cfg, on_retry=_default_on_retry)


def _default_on_retry(attempt: int, err: Exception) -> None:
    print(f"[retry] attempt={attempt} error={err}")


def _extract_text(resp: Any) -> str:
    """
    `google-genai` response objects may differ slightly across versions.
    Prefer `.text` when available; otherwise attempt common fallbacks.
    """
    if hasattr(resp, "text") and resp.text is not None:
        return str(resp.text)

    # Fallbacks
    candidates: list[Optional[str]] = []
    for attr in ("output_text", "candidates", "content"):
        if hasattr(resp, attr):
            candidates.append(str(getattr(resp, attr)))
    return "\n".join([c for c in candidates if c])


def _looks_retryable(e: Exception) -> bool:
    """
    Best-effort transient classification without importing provider-specific exception classes.
    """
    msg = (str(e) or "").lower()
    transient_markers = [
        "rate limit",
        "429",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "unavailable",
        "503",
        "500",
        "internal error",
        "connection reset",
        "connection error",
    ]
    return any(m in msg for m in transient_markers)


def _looks_permanent_quota_or_billing_issue(e: Exception) -> bool:
    """
    Heuristic for cases where retries won't help (e.g., your account/project has 0 quota).
    """
    msg = (str(e) or "").lower()
    return (
        ("resource_exhausted" in msg or "quota exceeded" in msg)
        and ("limit: 0" in msg or "quota value': '0" in msg or "quota_value': '0" in msg)
    )


def _summarize_error(e: Exception, max_len: int = 240) -> str:
    s = " ".join((str(e) or "").split())
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."

