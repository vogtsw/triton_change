from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]+"),
    re.compile(r"(?i)(api[_-]?key)(['\"\s:=]+)([^'\"\s]+)"),
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        sanitized = value
        for pattern in SECRET_PATTERNS:
            if "api" in pattern.pattern.lower():
                sanitized = pattern.sub(r"\1\2***REDACTED***", sanitized)
            else:
                sanitized = pattern.sub("sk-***REDACTED***", sanitized)
        return sanitized
    return value


def estimate_tokens(text: str) -> int:
    # Cheap, provider-agnostic estimate for fallback paths and snapshot accounting.
    return max(1, (len(text) + 3) // 4) if text else 0


class RunLogger:
    def __init__(self, log_dir: str | Path, run_id: str | None = None, verbose: bool = True):
        self.root = Path(log_dir)
        self.run_id = run_id or utc_stamp()
        self.dir = self.root / self.run_id
        self.verbose = verbose
        self.dir.mkdir(parents=True, exist_ok=True)
        self._counter = len(list(self.dir.glob("*.json")))

    def progress(self, message: str) -> None:
        if self.verbose:
            print(f"[langgraph] {message}", flush=True)

    def snapshot(self, kind: str, name: str, payload: Any) -> Path:
        self._counter += 1
        path = self.dir / f"{self._counter:02d}_{kind}_{name}.json"
        path.write_text(json.dumps(sanitize(payload), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def node_input(self, name: str, state: dict[str, Any]) -> None:
        self.progress(f"{name}: start")
        self.snapshot("node_input", name, state)

    def node_output(self, name: str, output: dict[str, Any]) -> None:
        self.snapshot("node_output", name, output)
        self.progress(f"{name}: done")

    def tool_call(self, name: str, payload: dict[str, Any]) -> None:
        self.snapshot("tool_call", name, payload)
        self.progress(f"tool {name}: {payload.get('action', 'called')}")

    def llm_call(self, name: str, payload: dict[str, Any]) -> None:
        self.snapshot("llm_call", name, payload)
        usage = payload.get("usage", {})
        if usage:
            self.progress(
                f"{name}: tokens in={usage.get('input_tokens', 0)} out={usage.get('output_tokens', 0)}"
            )


def logger_from_state(state: dict[str, Any]) -> RunLogger:
    return RunLogger(state.get("log_dir", "log"), state.get("run_id"))
