"""Shared task configuration types for generate_tasks / task_catalog."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_DEFAULT = {
    "hidden_size": 768,
    "intermediate_size": 3072,
    "seq_len": 128,
    "batch_size": 1,
    "ln_eps": 1e-5,
    "dtype_name": "float32",
}


def _fp(**kw: Any) -> dict[str, Any]:
    return {**_DEFAULT, **kw}


@dataclass
class TaskConfig:
    task_id: str
    tier: str
    profile: str
    change_types: list[str]
    description: str
    from_params: dict[str, Any]
    to_params: dict[str, Any]
    seed_offset: int = 0
    estimated_difficulty: int = 1
    dynamic_batch_target: bool = False

    @property
    def hidden_old(self) -> int:
        return int(self.from_params.get("hidden_size", 768))

    @property
    def hidden_new(self) -> int:
        return int(self.to_params.get("hidden_size", 768))

    @property
    def inter_old(self) -> int:
        return int(self.from_params.get("intermediate_size", 3072))

    @property
    def inter_new(self) -> int:
        return int(self.to_params.get("intermediate_size", 3072))

    @property
    def eps_old(self) -> float:
        return float(self.from_params.get("ln_eps", 1e-5))

    @property
    def eps_new(self) -> float:
        return float(self.to_params.get("ln_eps", 1e-5))

    @property
    def seq_old(self) -> int:
        return int(self.from_params.get("seq_len", 128))

    @property
    def seq_new(self) -> int:
        return int(self.to_params.get("seq_len", 128))

    @property
    def batch_old(self) -> int:
        return int(self.from_params.get("batch_size", 1))

    @property
    def batch_new(self) -> int:
        return int(self.to_params.get("batch_size", 1))

    @property
    def dtype_meta(self) -> str:
        name = str(self.to_params.get("dtype_name", "float32"))
        return {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}.get(name, "fp32")
