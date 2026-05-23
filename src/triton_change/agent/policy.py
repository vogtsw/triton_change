"""Agent policies — three flavors that share a thin interface.

The runner calls `policy.next_action(observation)` each step and gets back
a dict that conforms to the `agent_action` shape:

    {"tool": "apply_patch_ops",     "patch_ops": [...]}
    {"tool": "inspect_code_region", "region": "function:..."}
    {"tool": "run_static_check"}
    {"tool": "run_correctness_check"}
    {"tool": "finalize",            "reason": "..."}

DeepSeekPolicy uses the OpenAI-compatible chat API with response_format
JSON-object. OraclePolicy reads `oracle/patch_ops.json` and emits a single
apply_patch_ops + finalize. MockPolicy replays a queue of canned actions
(used in unit tests).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from triton_change.llm_clients import DeepSeekClient
from triton_change.agent.observation import Observation


__all__ = ["PolicyBase", "MockPolicy", "OraclePolicy", "DeepSeekPolicy"]


SYSTEM_PROMPT = """\
You are an autonomous Triton kernel migration agent. You will be given:

- An ONNX diff describing how `target.onnx` differs from `base.onnx` (raw
  diff and pre-extracted semantic_labels).
- A code summary of the current candidate file (constants, kernels,
  top-level functions, imports).
- The result of any prior tool call (failure_class, error tail, hint).

Each turn you must respond with EXACTLY ONE JSON object describing your
next action. Do not include any prose outside the JSON. Do not include
chain-of-thought.

ALLOWED ACTIONS

  {"tool": "apply_patch_ops", "patch_ops": [ ... ]}
  {"tool": "inspect_code_region", "region": "function:NAME" | "kernel:NAME" | "imports" | "constants" | "full"}
  {"tool": "run_static_check"}
  {"tool": "run_correctness_check"}
  {"tool": "finalize", "reason": "..."}

You normally do NOT need to call run_static_check or run_correctness_check
explicitly — after apply_patch_ops the runner runs them automatically and
feeds back results. Use them only to re-verify after inspecting code.

PATCH OP SCHEMA (operation, path, plus type-specific fields)

  update_constant      { constant_name, old_value?, new_value, reason? }
  update_kernel_meta   { kernel_name, meta_name, new_value, reason? }
  replace_function     { function_name, new_code, reason? }
  replace_kernel_body  { kernel_name, new_body, reason? }
  replace_region       { region, new_code, reason? }    region: function:NAME | kernel:NAME
  insert_after_region  { region, new_code, reason? }
  regex_replace        { pattern, replacement, count?, reason? }   (last resort)
  full_file_replace    { new_code, reason? }                       (heaviest penalty; avoid)

`path` MUST be exactly "candidate_model_triton.py". No "..", no leading "/".

OPERATIONAL GUIDANCE

- Pure shape / dim / epsilon constant changes  -> update_constant
- BLOCK_SIZE / num_warps / num_stages          -> update_kernel_meta
- Activation change (e.g. GELU -> ReLU)        -> replace_kernel_body or replace_function
- Norm change (LayerNorm -> RMSNorm)           -> replace_kernel_body
- Prefer the smallest set of surgical ops. Avoid full_file_replace.
- After applying patches, the runner WILL automatically run static + correctness
  and return the results in the next observation.
- If the same failure repeats, switch to a different op type or region.
- When correctness passes, emit {"tool": "finalize", "reason": "correctness pass"}.
"""


# ---------- Base ----------


class PolicyBase:
    name: str = "base"

    def next_action(self, observation: Observation) -> dict[str, Any]:
        raise NotImplementedError

    def call_log(self) -> list[dict[str, Any]]:
        return []

    def reset(self) -> None:
        pass


# ---------- Mock ----------


class MockPolicy(PolicyBase):
    """Replay a queue of canned actions. Loops the last action if the queue
    is exhausted (useful for tests that don't predict the exact step count).
    """

    name = "mock"

    def __init__(self, actions: Sequence[dict[str, Any]]):
        self._queue = list(actions)
        self._cursor = 0

    def next_action(self, observation: Observation) -> dict[str, Any]:
        if self._cursor < len(self._queue):
            action = self._queue[self._cursor]
            self._cursor += 1
            return action
        return {"tool": "finalize", "reason": "mock queue exhausted"}

    def reset(self) -> None:
        self._cursor = 0


# ---------- Oracle ----------


class OraclePolicy(PolicyBase):
    """Replay oracle/patch_ops.json. Step 0 = apply, step 1 = finalize."""

    name = "oracle"

    def __init__(self, task_dir: Path, patch_ops_path: Path | None = None):
        self.task_dir = Path(task_dir)
        self.patch_ops_path = patch_ops_path or (self.task_dir / "oracle" / "patch_ops.json")
        self._step = 0
        self._ops = json.loads(self.patch_ops_path.read_text(encoding="utf-8"))["ops"]

    def next_action(self, observation: Observation) -> dict[str, Any]:
        if self._step == 0:
            self._step += 1
            return {"tool": "apply_patch_ops", "patch_ops": self._ops}
        return {"tool": "finalize", "reason": "oracle done"}

    def reset(self) -> None:
        self._step = 0


# ---------- DeepSeek ----------


@dataclass
class _DSCallRecord:
    step_idx: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration_s: float
    raw_response: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_idx": self.step_idx,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "duration_s": round(self.duration_s, 3),
        }


class DeepSeekPolicy(PolicyBase):
    """LLM-driven policy using DeepSeek V3 (or compatible OpenAI client).

    Each call sends the system prompt + a single user message containing the
    current observation as JSON. JSON-object response format is requested so
    the model returns a parseable action.

    Temperature defaults to 0.7 on the first attempt and drops to 0.3 once a
    failure has been observed (per spec section 9).
    """

    name = "deepseek"

    def __init__(
        self,
        client: DeepSeekClient | None = None,
        *,
        model: str | None = None,
        temperature_initial: float = 0.7,
        temperature_retry: float = 0.3,
        max_tokens: int = 4096,
    ):
        self.client = client or DeepSeekClient()
        self.model = model or self.client.model
        self.t_initial = temperature_initial
        self.t_retry = temperature_retry
        self.max_tokens = max_tokens
        self._records: list[_DSCallRecord] = []

    def next_action(self, observation: Observation) -> dict[str, Any]:
        import time as _time

        temperature = self.t_initial if observation.last_error is None else self.t_retry
        user_msg = json.dumps({"observation": observation.to_dict()},
                              ensure_ascii=False)

        t0 = _time.time()
        resp = self.client.chat_json(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            model=self.model,
            temperature=temperature,
            max_tokens=self.max_tokens,
        )
        dt = _time.time() - t0

        # The chat_json wrapper returns the parsed dict already.
        # Extract usage from the last call record on the client.
        last_call = self.client.last_call_record()
        prompt_tokens = last_call.prompt_tokens if last_call else 0
        completion_tokens = last_call.completion_tokens if last_call else 0
        total_tokens = last_call.total_tokens if last_call else 0
        raw = last_call.raw_response if last_call else json.dumps(resp, ensure_ascii=False)

        self._records.append(_DSCallRecord(
            step_idx=observation.step_idx,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            duration_s=dt,
            raw_response=raw,
        ))

        return _normalize_action(resp)

    def call_log(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._records]

    def reset(self) -> None:
        self._records.clear()


def _normalize_action(action: dict[str, Any]) -> dict[str, Any]:
    """Coerce the raw LLM response into a clean action dict.

    Accepts both `{"tool": ...}` and `{"action": ...}` keys, strips a few
    common keys we do NOT want to record (e.g. `thinking`, `chain_of_thought`).
    """
    if not isinstance(action, dict):
        return {"tool": "finalize", "reason": f"invalid action (not dict): {action!r}"}

    tool = action.get("tool") or action.get("action") or ""
    out: dict[str, Any] = {"tool": tool}

    if tool == "apply_patch_ops":
        out["patch_ops"] = action.get("patch_ops") or action.get("ops") or []
    elif tool == "inspect_code_region":
        out["region"] = action.get("region", "full")
    elif tool == "finalize":
        out["reason"] = action.get("reason", "model requested finalize")
    # run_static_check / run_correctness_check / run_benchmark — no extra fields

    return out
