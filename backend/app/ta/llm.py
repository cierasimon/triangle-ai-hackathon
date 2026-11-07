"""OpenRouter chat client, free models only (plan.md section 3).

- Refuses any model id that doesn't end in ":free" and isn't "openrouter/free".
- These are reasoning models: max_tokens ≥ 1000, and an empty/null content is a failure
  (the budget went to reasoning).
- On 429/5xx/timeout/empty: try LLM_FALLBACK_MODEL once, then return None.
- With LLM_PROVIDER=none or no key, nothing is sent; callers use template mode on None.
Never call this in a loop, on every drag, or in tests (tests mock `chat`).
"""

import logging

import httpx

from app.config import get_settings

URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT_S = 8.0
MIN_MAX_TOKENS = 1000
log = logging.getLogger(__name__)

# tests may swap this for an httpx.MockTransport
transport: httpx.BaseTransport | None = None


class PaidModelError(ValueError):
    """A model id that is not on the free tier."""


def check_free(model: str) -> str:
    if model == "openrouter/free" or model.endswith(":free"):
        return model
    raise PaidModelError(f"refusing non-free model {model!r}")


def _call(model: str, messages: list[dict], max_tokens: int, key: str) -> str | None:
    """One request. Returns text, or None on a retryable/empty failure."""
    check_free(model)
    try:
        with httpx.Client(timeout=TIMEOUT_S, transport=transport) as client:
            r = client.post(URL, headers={"Authorization": f"Bearer {key}",
                                          "X-Title": "PaperPhysics"},
                            json={"model": model, "messages": messages,
                                  "max_tokens": max(max_tokens, MIN_MAX_TOKENS)})
    except httpx.HTTPError as exc:
        log.warning("LLM %s: %s", model, type(exc).__name__)
        return None
    if r.status_code != 200:
        log.warning("LLM %s: HTTP %s", model, r.status_code)
        return None
    try:
        content = r.json()["choices"][0]["message"].get("content")
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    if not content or not str(content).strip():
        log.warning("LLM %s: empty content (reasoning used the budget?)", model)
        return None
    return str(content).strip()


def chat(messages: list[dict], *, model: str | None = None, max_tokens: int = 1500,
         vision: bool = False) -> str | None:
    s = get_settings()
    if not s.llm_enabled:
        return None
    primary = model or (s.llm_vision_model if vision else s.llm_model)
    # validate both before sending anything
    check_free(primary)
    check_free(s.llm_fallback_model)
    text = _call(primary, messages, max_tokens, s.openrouter_api_key)
    if text is None and s.llm_fallback_model != primary:
        text = _call(s.llm_fallback_model, messages, max_tokens, s.openrouter_api_key)
    return text
