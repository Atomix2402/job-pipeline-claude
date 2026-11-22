"""LLM helper: Groq + OpenRouter, with per-task routing and health probing.

Public API:
  complete_json(system, prompt, task="read") -> dict
  refresh_health(provider=None, full=False)
Diagnostic:  uv run llm/provider.py   (probes every configured model)

Env vars (GitHub Secrets):
  GROQ_API_KEY
  OPENROUTER_API_KEY
Env vars (plain):
  OPENROUTER_MODELS   optional override, comma list (free ids end in ":free");
                      default: nemotron-3.5-lightning, ling-3.2-flash, ernie-5.5-flash-preview,
                      nemotron-3-ultra-550b-a55b
  GROQ_MODELS         default "openai/gpt-oss-20b,openai/gpt-oss-120b" (cheapest first)
  LLM_ROUTE_READ      default "groq,openrouter"      (task: read the email, match the job)
  LLM_ROUTE_CLASSIFY  default "openrouter,groq"      (task: classify the recruitment event)
  MODEL_TIMEOUT       default 90   seconds to wait for one model's real answer
  PROBE_TIMEOUT       default 12   seconds allowed for the tiny health probe
  HEALTHY_TARGET      default 1    healthy models to find per provider (1 = one active model each)

Each task tries its providers in order; inside a provider, requests rotate across
healthy models, and a failing model is dropped until the next re-probe.
Free tiers may use prompts to improve models, so callers send minimal, redacted text.
"""
import json
import os
import re
import time

import requests

CONNECT_TIMEOUT = 5
MODEL_TIMEOUT = float(os.getenv("MODEL_TIMEOUT", "90"))
PROBE_TIMEOUT = float(os.getenv("PROBE_TIMEOUT", "12"))
HEALTHY_TARGET = int(os.getenv("HEALTHY_TARGET", "1"))   # 1 = pick a single model per provider
REPROBE_SECONDS = 600

PROVIDERS = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": "GROQ_API_KEY",
        "models_env": "GROQ_MODELS",
        "default_models": "openai/gpt-oss-20b,openai/gpt-oss-120b",
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
        "models_env": "OPENROUTER_MODELS",
        # fast models first; the 550B reasoning model last (strong but slower)
        "default_models": ("nvidia/nemotron-3.5-lightning:free,inclusionai/ling-3.2-flash:free,"
                           "baidu/ernie-5.5-flash-preview:free,nvidia/nemotron-3-ultra-550b-a55b:free"),
    },
}
DEFAULT_ROUTES = {"read": "groq,openrouter", "classify": "openrouter,groq"}

PROBE_SYSTEM = "Reply with JSON only."
PROBE_PROMPT = 'Return exactly: {"ok": true}'
UNAVAILABLE = ("network", "rate_limited", "overloaded")   # model down right now, not a bad answer


class LLMError(Exception):
    pass


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise LLMError("bad_response: not JSON")


def _post(url, headers, body, read_timeout):
    """Single attempt. Keys travel in headers, never in URLs/logs."""
    try:
        resp = requests.post(url, headers=headers, json=body,
                             timeout=(CONNECT_TIMEOUT, read_timeout))
    except requests.RequestException as e:
        raise LLMError(f"network:{type(e).__name__}")
    if resp.status_code == 429:
        raise LLMError("rate_limited")
    if resp.status_code >= 500:
        raise LLMError(f"overloaded:{resp.status_code}")
    if resp.status_code != 200:
        raise LLMError(f"HTTP {resp.status_code}: {resp.text[:150]}")
    return resp.json()


def _call(provider, model, system, prompt, timeout):
    """Both providers speak the OpenAI chat-completions format."""
    cfg = PROVIDERS[provider]
    headers = {"Authorization": f"Bearer {os.getenv(cfg['key_env'], '')}",
               "Content-Type": "application/json"}
    body = {"model": model, "temperature": 0,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}]}
    for json_mode in (True, False):   # some models reject JSON mode (HTTP 400)
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        else:
            body.pop("response_format", None)
        try:
            data = _post(cfg["url"], headers, body, timeout)
            return _parse_json(data["choices"][0]["message"]["content"])
        except LLMError as e:
            if json_mode and str(e).startswith("HTTP 400"):
                continue
            raise
        except (KeyError, IndexError, TypeError):
            raise LLMError("bad_response: unexpected shape")


def _models(provider) -> list[str]:
    cfg = PROVIDERS[provider]
    if not os.getenv(cfg["key_env"]):
        return []
    raw = os.getenv(cfg["models_env"], cfg["default_models"])
    return [m.strip() for m in raw.split(",") if m.strip()]


_state = {"healthy": {}, "probed_at": {}, "rr": {}, "errors": {}}


def _probe(provider, model):
    start = time.monotonic()
    try:
        result = _call(provider, model, PROBE_SYSTEM, PROBE_PROMPT, PROBE_TIMEOUT)
    except LLMError as e:
        return False, str(e)[:80]
    if not isinstance(result, dict):
        return False, "bad_response"
    return True, f"{int((time.monotonic() - start) * 1000)}ms"


def refresh_health(provider=None, full=False) -> dict:
    """Probe models with a tiny request. Stops at HEALTHY_TARGET per provider unless full."""
    for prov in ([provider] if provider else list(PROVIDERS)):
        models = _models(prov)
        healthy, errors = [], []
        if not models:
            reason = "no key set" if not os.getenv(PROVIDERS[prov]["key_env"]) else \
                     f"{PROVIDERS[prov]['models_env']} not set"
            print(f"probe {prov}: skipped ({reason})")
        for model in models:
            ok, detail = _probe(prov, model)
            print(f"probe {prov}/{model}: {'ok ' + detail if ok else 'FAIL ' + detail}")
            if ok:
                healthy.append(model)
                if not full and len(healthy) >= HEALTHY_TARGET:
                    break
            else:
                errors.append(f"{model}: {detail}")
        if healthy:
            print(f"selected {prov}/{healthy[0]}")
        _state["healthy"][prov] = healthy
        _state["errors"][prov] = errors
        _state["probed_at"][prov] = time.monotonic()
    return _state["healthy"]


def _call_provider(provider, system, prompt) -> dict:
    """Rotate across this provider's healthy models; re-probe once if all fail."""
    stale = time.monotonic() - _state["probed_at"].get(provider, 0) > REPROBE_SECONDS
    if provider not in _state["healthy"] or stale:
        refresh_health(provider)

    errors: list[str] = []
    for round_no in range(2):
        healthy = list(_state["healthy"].get(provider, []))
        if healthy:
            start = _state["rr"].get(provider, 0) % len(healthy)
            _state["rr"][provider] = _state["rr"].get(provider, 0) + 1
            for model in healthy[start:] + healthy[:start]:
                try:
                    return _call(provider, model, system, prompt, MODEL_TIMEOUT)
                except LLMError as e:
                    errors.append(f"{model}: {str(e)[:60]}")
                    if str(e).startswith(UNAVAILABLE) and model in _state["healthy"][provider]:
                        _state["healthy"][provider].remove(model)
        if round_no == 0:
            refresh_health(provider)

    errors += _state["errors"].get(provider, [])
    raise LLMError("; ".join(errors) or "no usable models")


def complete_json(system: str, prompt: str, task: str = "read") -> dict:
    """Run one task on its provider route (e.g. read -> groq first, classify -> openrouter first)."""
    route = os.getenv(f"LLM_ROUTE_{task.upper()}", DEFAULT_ROUTES.get(task, "groq,openrouter"))
    providers = [p.strip() for p in route.split(",") if p.strip() in PROVIDERS]

    errors = []
    for provider in providers:
        if not _models(provider):
            errors.append(f"{provider}: not configured")
            continue
        try:
            return _call_provider(provider, system, prompt)
        except LLMError as e:
            errors.append(f"{provider}: {e}")
    raise LLMError(f"[{task}] " + " | ".join(errors))


if __name__ == "__main__":
    print("Probing every configured model...\n")
    result = refresh_health(full=True)
    total = sum(len(v) for v in result.values())
    print(f"\n{total} healthy models: " + ", ".join(f"{p}={len(m)}" for p, m in result.items()))