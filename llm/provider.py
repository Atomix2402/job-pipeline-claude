"""Provider-agnostic LLM helper. One function: complete_json().

Env vars (GitHub Secrets):
  GEMINI_API_KEY   - Google AI Studio key (free tier; do NOT enable billing)
  GROQ_API_KEY     - optional fallback (free tier)
Optional env (plain variables, not secrets):
  LLM_PROVIDER     - "gemini" (default) or "groq"
  GEMINI_MODEL     - default "gemini-2.5-flash" (check AI Studio for current free models)
  GROQ_MODEL       - default "llama-3.3-70b-versatile"

Free tiers can use your prompts to improve models, so callers should send
only the minimum email text needed (no phone numbers, addresses, OTPs).
"""
import json
import os
import re
import time

import requests

TIMEOUT = 60


class LLMError(Exception):
    pass


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise LLMError("Model did not return valid JSON")


def _post(url, headers, body, retries=3):
    """POST with retry on 429/5xx. Keys travel in headers, never in URLs/logs."""
    for attempt in range(retries):
        resp = requests.post(url, headers=headers, json=body, timeout=TIMEOUT)
        if resp.status_code == 429:
            if attempt == retries - 1:
                raise LLMError("rate_limited")
            time.sleep(5 * (attempt + 1))
            continue
        if resp.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        break
    if resp.status_code != 200:
        raise LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.json()


def _gemini(system: str, prompt: str) -> dict:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY not set")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    data = _post(url, {"x-goog-api-key": key, "Content-Type": "application/json"}, body)
    try:
        return _parse_json(data["candidates"][0]["content"]["parts"][0]["text"])
    except (KeyError, IndexError):
        raise LLMError("Unexpected Gemini response shape")


def _groq(system: str, prompt: str) -> dict:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY not set")
    body = {
        "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    data = _post("https://api.groq.com/openai/v1/chat/completions",
                 {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                 body)
    try:
        return _parse_json(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError):
        raise LLMError("Unexpected Groq response shape")


_PROVIDERS = {"gemini": _gemini, "groq": _groq}


def complete_json(system: str, prompt: str) -> dict:
    """Return the model's JSON answer as a dict.

    Uses LLM_PROVIDER first; if it fails (rate limit, outage) and the other
    provider's key exists, falls back to it. Raises LLMError if all fail.
    """
    primary = os.getenv("LLM_PROVIDER", "gemini").lower()
    if primary not in _PROVIDERS:
        raise LLMError(f"Unknown LLM_PROVIDER: {primary}")
    order = [primary] + [p for p in _PROVIDERS if p != primary]

    errors = []
    for name in order:
        try:
            return _PROVIDERS[name](system, prompt)
        except LLMError as e:
            errors.append(f"{name}: {e}")
    raise LLMError("; ".join(errors))