"""One tiny LLM client for every free provider, with an offline fallback.

Every call takes a `mock` callable. If the provider has no key, is down, or LLM_MODE=mock,
the mock result is returned instead, so the demo never dies on stage.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable

from . import config

log = logging.getLogger("ai_layer.llm")
_clients: dict[str, Any] = {}


def _parse_spec(spec: str) -> tuple[str, str]:
    provider, _, model = spec.partition(":")
    if provider not in config.PROVIDERS or not model:
        raise ValueError(f"Bad model spec '{spec}'. Use provider:model, e.g. groq:llama-3.1-8b-instant")
    return provider, model


def provider_available(spec: str) -> bool:
    if config.LLM_MODE == "mock":
        return False
    provider, _ = _parse_spec(spec)
    key_env = config.PROVIDERS[provider]["key_env"]
    if key_env is None:  # ollama: assume available only if user opted in
        return os.getenv("USE_OLLAMA", "0") == "1"
    return bool(os.getenv(key_env))


def _client(provider: str):
    if provider not in _clients:
        from openai import OpenAI

        cfg = config.PROVIDERS[provider]
        key = os.getenv(cfg["key_env"]) if cfg["key_env"] else "ollama"
        _clients[provider] = OpenAI(base_url=cfg["base_url"], api_key=key, timeout=60)
    return _clients[provider]


def chat(
    messages: list[dict],
    model: str,
    mock: Callable[[], str],
    temperature: float = 0.2,
    max_tokens: int = 3000,
) -> tuple[str, str]:
    """Returns (text, source). source is the model spec, or 'mock'."""
    if not provider_available(model):
        return mock(), "mock"
    provider, name = _parse_spec(model)
    try:
        kw = {}
        if "gpt-oss" in name or "qwen3" in name:  # reasoning models: keep thinking short so the answer fits
            kw["extra_body"] = {"reasoning_effort": "low"}
        resp = _client(provider).chat.completions.create(
            model=name, messages=messages, temperature=temperature, max_tokens=max_tokens, **kw
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            log.warning("LLM %s returned empty content (reasoning ate the budget?); using mock", model)
            return mock(), "mock"
        return text, model
    except Exception as e:  # network, rate limit, bad model name...
        log.warning("LLM call to %s failed (%s); using mock", model, e)
        return mock(), "mock"


def extract_json(text: str) -> Any:
    """Pull the first JSON object/array out of a model reply (handles ```json fences)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return json.loads(text)


def chat_json(messages: list[dict], model: str, mock: Callable[[], Any], raw_key: str | None = None, **kw) -> tuple[Any, str]:
    """Like chat(), but returns parsed JSON. If parsing fails: with raw_key, returns {raw_key: text} so the
    caller can still use the model's real answer; otherwise falls back to mock."""
    text, source = chat(messages, model, mock=lambda: json.dumps(mock()), **kw)
    try:
        return extract_json(text), source
    except Exception:
        if raw_key and source != "mock" and text.strip():
            log.warning("Could not parse JSON from %s; using its raw text", source)
            return {raw_key: text.strip()}, source
        log.warning("Could not parse JSON from %s; using mock", source)
        return mock(), "mock"
