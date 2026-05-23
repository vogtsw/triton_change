"""Dump a specific step from a trajectory for debugging."""
import json, sys
from pathlib import Path

p = Path(sys.argv[1])
idx = int(sys.argv[2]) if len(sys.argv) > 2 else 1

t = json.loads(p.read_text(encoding="utf-8"))
s = t["steps"][idx]
print("=== STEP", idx, "ACTION ===")
print(json.dumps(s["action"], indent=2, ensure_ascii=False))
print()
print("=== STEP", idx, "TOOL_RESULTS ===")
for r in s["tool_result"].get("results", []):
    print("tool=", r.get("tool"), "success=", r.get("success"), "fc=", r.get("failure_class"))
    if not r.get("success"):
        print("  error=", (r.get("error") or "")[:600])
