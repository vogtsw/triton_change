"""Quick CLI to summarize a saved trajectory.json."""
from __future__ import annotations
import json, sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else "tasks/task_000001/trajectory.json")
t = json.loads(p.read_text(encoding="utf-8"))
print(f"task_id={t['task_id']} success={t['success']} final_reward={t['final_reward']:.3f}")
print(f"agent_model={t.get('agent_model')} steps={t['total_steps']} patch_ops={t['total_patch_ops']}")
print()
for s in t["steps"]:
    tool = s["action"].get("tool", "?")
    fc = s.get("failure_class")
    print(f"--- step {s['step_idx']}: tool={tool} reward={s['step_reward']:+.3f} failure_class={fc} ---")
    if tool == "apply_patch_ops":
        for i, op in enumerate(s["action"].get("patch_ops", [])):
            preview = {k: (v if not isinstance(v, str) or len(v) < 80 else v[:80] + "...") for k, v in op.items()}
            print(f"  op {i}: {preview}")
    elif tool == "inspect_code_region":
        print(f"  region={s['action'].get('region')}")
    elif tool == "finalize":
        print(f"  reason={s['action'].get('reason')}")
    if fc:
        results = s["tool_result"].get("results", [])
        for r in results:
            if not r.get("success"):
                err = (r.get("error") or "")[:300]
                print(f"  ERR: {err}")
                break
print()
print("llm_call_log summary:")
for c in t.get("llm_call_log", []):
    print(f"  step {c['step_idx']}: {c['total_tokens']} tok, {c['duration_s']:.2f}s")
