"""Thin, provider-agnostic LLM client (Google / Anthropic / OpenAI) over plain REST.

Reliability features:
  * on-disk response cache  -> the same input always gives the same output across runs
  * request throttling      -> stays under the free-tier requests-per-minute limit
  * retry with backoff      -> survives 429 / 5xx errors, honours Retry-After
  * robust JSON extraction  -> tolerates code fences and stray text around the JSON
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import requests

from .config import Config
from .imaging import BillImage

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the model cannot return a usable response."""


class QuotaExceededError(LLMError):
    """Raised when a hard (e.g. daily) quota is exhausted; retrying will not help."""


def extract_json(text: str) -> Any:
    """Parse a JSON value out of model text, tolerating ```json fences and surrounding prose."""
    if text is None:
        raise ValueError("empty response")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1]
    if not starts:
        raise ValueError("no JSON found in model response")
    start = min(starts)
    end = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass
    return _lenient_loads(cleaned[start:])


def _lenient_loads(text: str) -> Any:
    """Last-resort repair: drop trailing commas and close a reply that was cut off mid-way."""
    text = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    stack: list[str] = []
    in_string = escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    repaired = text + ('"' if in_string else "")
    repaired = re.sub(r",\s*$", "", repaired) + "".join(reversed(stack))
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in model response: {exc}") from exc


class LLMClient:
    def __init__(self, config: Config):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.use_cache = True
        self.calls_made = 0
        self.cache_hits = 0
        self._last_call = 0.0
        self._session = requests.Session()

    # ------------------------------------------------------------------ public API
    def generate_json(self, system: str, prompt: str, images: list[BillImage] | None = None,
                      max_output_tokens: int = 8192, label: str = "") -> Any:
        """Call the model and return parsed JSON. Cached by (model, prompts, images)."""
        images = images or []
        key = self._cache_key(system, prompt, images, max_output_tokens)
        cached = self._read_cache(key)
        if cached is not None:
            self.cache_hits += 1
            log.info("cache hit  %s", label)
            try:
                return extract_json(cached)
            except ValueError:
                log.warning("cached response for %s was not valid JSON; refetching", label)

        text = self._call_with_retries(system, prompt, images, max_output_tokens, label)
        try:
            parsed = extract_json(text)
        except ValueError as exc:
            log.warning("%s: reply was not valid JSON (%s); asking once more", label, exc)
            nudge = ("\n\nIMPORTANT: your previous reply was not valid JSON. Reply again with ONE complete, valid "
                     "JSON object only: no markdown, no comments, no trailing commas, and never put double quotes "
                     "inside a string (use single quotes instead).")
            text = self._call_with_retries(system, prompt + nudge, images, max_output_tokens, f"{label} (json retry)")
            parsed = extract_json(text)      # still invalid -> ValueError, caller decides how to recover
        self._write_cache(key, text)         # only cache responses that parse
        return parsed

    # ------------------------------------------------------------------ caching
    def _cache_key(self, system: str, prompt: str, images: list[BillImage], max_tokens: int) -> str:
        h = hashlib.sha256()
        for part in (self.config.provider, self.config.model, str(self.config.temperature),
                     str(self.config.seed), str(max_tokens), system, prompt):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        for img in images:
            h.update(hashlib.sha256(img.data).digest())
        return h.hexdigest()

    def _read_cache(self, key: str) -> str | None:
        if not self.use_cache:
            return None
        path = self.cache_dir / f"{key}.txt"
        try:
            return path.read_text(encoding="utf-8") if path.exists() else None
        except OSError:
            return None

    def _write_cache(self, key: str, text: str) -> None:
        if not self.use_cache:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / f"{key}.txt").write_text(text, encoding="utf-8")
        except OSError as exc:  # a read-only disk must not break the run
            log.warning("could not write cache: %s", exc)

    # ------------------------------------------------------------------ transport
    def _throttle(self) -> None:
        min_gap = 60.0 / max(self.config.requests_per_minute, 0.1)
        wait = self._last_call + min_gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _call_with_retries(self, system, prompt, images, max_tokens, label) -> str:
        last_error = "unknown error"
        for attempt in range(self.config.max_retries + 1):
            self._throttle()
            try:
                self.calls_made += 1
                log.info("model call %s (attempt %d)", label, attempt + 1)
                return self._send(system, prompt, images, max_tokens)
            except requests.RequestException as exc:
                last_error, delay = f"network error: {exc}", min(2 ** (attempt + 1), 30)
            except _HttpError as exc:
                last_error = f"HTTP {exc.status}: {exc.body[:300]}"
                if exc.status in (400, 401, 403, 404):
                    raise LLMError(f"{last_error} (check API key, model id and request)") from exc
                if exc.status == 429 and _is_daily_quota(exc.body):
                    raise QuotaExceededError(f"Daily quota exhausted: {exc.body[:300]}") from exc
                delay = exc.retry_after or min(2 ** (attempt + 1), 60)
            if attempt < self.config.max_retries:
                log.warning("%s -> retrying in %.0fs", last_error, delay)
                time.sleep(delay)
        raise LLMError(f"Giving up after {self.config.max_retries + 1} attempts: {last_error}")

    def _send(self, system: str, prompt: str, images: list[BillImage], max_tokens: int) -> str:
        provider = self.config.provider
        if provider == "google":
            return self._send_google(system, prompt, images, max_tokens)
        if provider == "anthropic":
            return self._send_anthropic(system, prompt, images, max_tokens)
        if provider == "openai":
            return self._send_openai(system, prompt, images, max_tokens)
        raise LLMError(f"Unsupported provider {provider}")

    def _post(self, url: str, headers: dict, body: dict) -> dict:
        resp = self._session.post(url, headers=headers, json=body, timeout=self.config.timeout)
        if resp.status_code != 200:
            retry_after = None
            try:
                retry_after = float(resp.headers.get("retry-after", ""))
            except ValueError:
                pass
            raise _HttpError(resp.status_code, resp.text, retry_after)
        return resp.json()

    def _send_google(self, system, prompt, images, max_tokens) -> str:
        cfg = self.config
        parts: list[dict] = [{"text": prompt}]
        for img in images:
            parts.append({"inline_data": {"mime_type": img.mime_type,
                                          "data": base64.b64encode(img.data).decode()}})
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": cfg.temperature,
                "seed": cfg.seed,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            },
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{cfg.model}:generateContent"
        data = self._post(url, {"x-goog-api-key": cfg.api_key, "Content-Type": "application/json"}, body)
        candidates = data.get("candidates") or []
        if not candidates:
            raise LLMError(f"Gemini returned no candidates: {json.dumps(data)[:300]}")
        parts_out = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts_out if not p.get("thought"))
        if not text:
            raise LLMError(f"Gemini returned empty text (finishReason={candidates[0].get('finishReason')})")
        return text

    def _send_anthropic(self, system, prompt, images, max_tokens) -> str:
        cfg = self.config
        content: list[dict] = []
        for img in images:
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": img.mime_type,
                "data": base64.b64encode(img.data).decode()}})
        content.append({"type": "text", "text": prompt})
        body = {
            "model": cfg.model,
            "max_tokens": max_tokens,
            "temperature": cfg.temperature,
            "system": system,
            "messages": [{"role": "user", "content": content}],
        }
        headers = {"x-api-key": cfg.api_key, "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        data = self._post("https://api.anthropic.com/v1/messages", headers, body)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        if not text:
            raise LLMError("Claude returned empty text")
        return text

    def _send_openai(self, system, prompt, images, max_tokens) -> str:
        cfg = self.config
        content: list[dict] = [{"type": "text", "text": prompt}]
        for img in images:
            b64 = base64.b64encode(img.data).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{img.mime_type};base64,{b64}"}})
        body: dict[str, Any] = {
            "model": cfg.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "response_format": {"type": "json_object"},
            "seed": cfg.seed,
        }
        if cfg.model.startswith("gpt-5"):
            # Reasoning models reject temperature; keep reasoning small so output tokens go to the answer.
            body["max_completion_tokens"] = max_tokens + 4096
            body["reasoning_effort"] = "low"
        else:
            body["max_tokens"] = max_tokens
            body["temperature"] = cfg.temperature
        headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
        data = self._post("https://api.openai.com/v1/chat/completions", headers, body)
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        if not text:
            raise LLMError("OpenAI returned empty text")
        return text


class _HttpError(Exception):
    def __init__(self, status: int, body: str, retry_after: float | None = None):
        super().__init__(f"HTTP {status}")
        self.status, self.body, self.retry_after = status, body, retry_after


def _is_daily_quota(body: str) -> bool:
    lowered = body.lower()
    return "perday" in lowered.replace(" ", "").replace("_", "") or "requests per day" in lowered