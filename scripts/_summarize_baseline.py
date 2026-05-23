"""Summarize the latest N entries in a baseline jsonl file."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
path = REPO / "trajectories" / "baseline_deepseek.jsonl"
n = int(sys.argv[1]) if len(sys.argv) > 1 else 20

lines = path.read_text(encoding="utf-8").strip().splitlines()
rows = [json.loads(l) for l in lines[-n:]]

fc = Counter(r.get("failure_class") or "none" for r in rows)
print(f"=== Latest {len(rows)}-task DeepSeek baseline ===")
print(f"success: {sum(r['success'] for r in rows)} / {len(rows)}")
print(f"failure_class: {dict(fc)}")
print(f"avg_reward: {sum(r['final_reward'] for r in rows) / len(rows):.3f}")
print(f"avg_steps: {sum(r['total_steps'] for r in rows) / len(rows):.2f}")
tok = sum(sum(c.get("total_tokens", 0) for c in r.get("llm_call_log", [])) for r in rows)
print(f"total_tokens: {tok}")
print()
for r in rows:
    steps = r["steps"]
    first_patch = next((s for s in steps if s["action"].get("tool") == "apply_patch_ops"), None)
    ops = (first_patch or {}).get("action", {}).get("patch_ops", [])
    parts = []
    for op in ops[:4]:
        if op.get("operation") == "update_constant":
            parts.append(f"{op.get('constant_name')}={op.get('new_value')}")
        else:
            parts.append(str(op.get("operation", "?")))
    summary = "; ".join(parts) if parts else "none"
    print(f"{r['task_id']}: fc={r.get('failure_class')} steps={r['total_steps']} first_patch=[{summary}]")
