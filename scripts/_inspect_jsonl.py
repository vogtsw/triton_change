"""Inspect a specific task's trajectory in a JSONL baseline log."""
import json, sys
from pathlib import Path

p = Path(sys.argv[1])
target_task = sys.argv[2] if len(sys.argv) > 2 else None

with open(p, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        t = json.loads(line)
        if target_task and t["task_id"] != target_task:
            continue
        print(f"\n### {t['task_id']} (success={t['success']} reward={t['final_reward']:.3f} steps={t['total_steps']})")
        for s in t["steps"]:
            tool = s["action"].get("tool", "?")
            fc = s.get("failure_class")
            print(f"  step {s['step_idx']}: tool={tool:<22s} reward={s['step_reward']:+.3f} fc={fc}")
            obs = s.get("observation", {})
            if obs.get("hint"):
                print(f"    hint: {obs['hint'][:140]}{'...' if len(obs['hint']) > 140 else ''}")
            if obs.get("repeated_same_error"):
                print(f"    repeated_same_error=True")
            if tool == "apply_patch_ops":
                ops = s["action"].get("patch_ops", [])
                for i, op in enumerate(ops):
                    op_type = op.get("operation")
                    if op_type == "update_constant":
                        print(f"    op {i}: update_constant {op.get('constant_name')} -> {op.get('new_value')}")
                    elif op_type == "full_file_replace":
                        print(f"    op {i}: FULL_FILE_REPLACE ({len(op.get('new_code',''))} chars)")
                    else:
                        print(f"    op {i}: {op_type}")
            elif tool == "finalize":
                print(f"    reason: {s['action'].get('reason', '')[:140]}")
            elif tool == "inspect_code_region":
                print(f"    region: {s['action'].get('region')}")
        if target_task:
            break
