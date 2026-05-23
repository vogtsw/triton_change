"""Trajectory writers — JSONL (one trajectory per line) and pretty JSON.

Conforms to `schemas/trajectory_schema.json`. The schema's required fields are
all present on AgentRunResult; this module just owns the serialization concerns:

- Append-mode JSONL for batched RL data.
- Compact-but-readable indented JSON for one-off runs.
- Optional schema validation pass via jsonschema.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


__all__ = ["write_trajectory_jsonl", "write_trajectory_pretty",
           "load_trajectory_schema", "validate_trajectory"]


def write_trajectory_jsonl(trajectories: Iterable[dict[str, Any]], path: Path) -> None:
    """Append-write trajectories as one JSON object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for tr in trajectories:
            f.write(json.dumps(tr, ensure_ascii=False))
            f.write("\n")


def write_trajectory_pretty(trajectory: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(trajectory, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_trajectory_schema() -> dict[str, Any]:
    here = Path(__file__).resolve()
    repo = here.parents[3]
    schema_path = repo / "schemas" / "trajectory_schema.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


def validate_trajectory(trajectory: dict[str, Any]) -> None:
    """Raise jsonschema.ValidationError if invalid; no-op if jsonschema absent."""
    try:
        from jsonschema import validate
    except ImportError:
        return
    validate(trajectory, load_trajectory_schema())
