"""DeepSeek API client (OpenAI-compatible).

DeepSeek's Chat Completions API is a drop-in for OpenAI's. We wrap it with:

- Auto-loaded .env (best effort, no hard dep on python-dotenv).
- Retries with exponential backoff.
- Per-instance call log usable for trajectory accounting.
- Both `deepseek-chat` (V3, default) and `deepseek-reasoner` are supported via
  the `model` argument or DEEPSEEK_MODEL env var.

Environment variables (in priority order):

    DEEPSEEK_API_KEY > OPENAI_API_KEY      # API key
    DEEPSEEK_BASE_URL                       # default https://api.deepseek.com/v1
    DEEPSEEK_MODEL                          # default deepseek-chat

Never commit the key. Always load it from .env or process env.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI


def _maybe_load_dotenv() -> None:
    """Best-effort .env load. Silently no-op if python-dotenv is missing."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        return
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        env_file = parent / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            return


@dataclass
class CallRecord:
    """One record per successful chat call. Used for trajectory accounting."""

    model: str
    elapsed_s: float
    attempt: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "elapsed_s": self.elapsed_s,
            "attempt": self.attempt,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class DeepSeekClient:
    """Thin wrapper over the OpenAI Python SDK pointed at DeepSeek."""

    DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
    DEFAULT_MODEL = "deepseek-chat"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        _maybe_load_dotenv()
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY (or OPENAI_API_KEY) not set. "
                "Copy .env.example to .env and fill in your key."
            )
        base_url = (
            base_url
            or os.getenv("DEEPSEEK_BASE_URL")
            or self.DEFAULT_BASE_URL
        )
        model = model or os.getenv("DEEPSEEK_MODEL") or self.DEFAULT_MODEL

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.call_log: list[CallRecord] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict[str, str] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call chat completions with retries; return content + usage."""
        last_err: Exception | None = None
        # Drop any sneaky model kwarg in **kwargs (would clash below)
        kwargs.pop("model", None)
        chosen_model = model or self.model
        for attempt in range(self.max_retries):
            try:
                start = time.time()
                resp = self.client.chat.completions.create(
                    model=chosen_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    **kwargs,
                )
                elapsed = time.time() - start
                content = resp.choices[0].message.content or ""
                usage = resp.usage
                rec = CallRecord(
                    model=chosen_model,
                    elapsed_s=elapsed,
                    attempt=attempt,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                    total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
                    raw_response=content,
                )
                self.call_log.append(rec)
                return {
                    "content": content,
                    "elapsed_s": elapsed,
                    "usage": {
                        "prompt_tokens": rec.prompt_tokens,
                        "completion_tokens": rec.completion_tokens,
                        "total_tokens": rec.total_tokens,
                    },
                    "attempt": attempt,
                }
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    time.sleep(wait)
        raise RuntimeError(
            f"DeepSeek call failed after {self.max_retries} retries: {last_err}"
        )

    def chat_json(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Convenience wrapper that forces JSON output and returns the parsed object.

        Falls back to a substring extraction if the model wraps the JSON in
        prose despite `response_format`. If parsing fails entirely, returns
        a {"_parse_error": ..., "_raw": ...} so callers can decide how to react.
        """
        import json as _json

        kwargs.setdefault("response_format", {"type": "json_object"})
        kwargs.setdefault("temperature", 0.3)
        result = self.chat(messages, **kwargs)
        content = result["content"]
        try:
            return _json.loads(content)
        except _json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if 0 <= start < end:
                try:
                    return _json.loads(content[start : end + 1])
                except _json.JSONDecodeError:
                    pass
            return {"_parse_error": "invalid JSON in response", "_raw": content}

    def last_call_record(self) -> CallRecord | None:
        return self.call_log[-1] if self.call_log else None

    def total_usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": sum(r.prompt_tokens for r in self.call_log),
            "completion_tokens": sum(r.completion_tokens for r in self.call_log),
            "total_tokens": sum(r.total_tokens for r in self.call_log),
            "calls": len(self.call_log),
        }
