"""Export trajectories to SFT / DPO / RL dataset formats (data prep only, no training).

Usage:
    python scripts/export_trajectory.py sft trajectories/baseline_oracle.jsonl
    python scripts/export_trajectory.py dpo data/dpo_pairs.jsonl
    python scripts/export_trajectory.py rl trajectories/baseline_oracle.jsonl
    python scripts/export_trajectory.py all trajectories/baseline_oracle.jsonl -o data/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from export_sft import extract_sft_pairs, export_sft  # noqa: E402


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def export_dpo(input_path: Path, output_path: Path) -> int:
    """DPO format: prompt + chosen + rejected patch_ops."""
    n = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for row in _iter_jsonl(input_path):
            if "chosen" not in row or "rejected" not in row:
                continue
            record = {
                "task_id": row.get("task_id"),
                "prompt": f"Migrate Triton kernel for {row.get('task_id')}",
                "chosen": json.dumps(row["chosen"].get("patch_ops", row["chosen"]), ensure_ascii=False),
                "rejected": json.dumps(row["rejected"].get("patch_ops", row["rejected"]), ensure_ascii=False),
                "margin": row.get("margin"),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def export_rl(input_path: Path, output_path: Path) -> int:
    """RL prep format: full trajectory + scalar reward (no advantage computation)."""
    n = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for traj in _iter_jsonl(input_path):
            record = {
                "task_id": traj.get("task_id"),
                "tier": traj.get("tier"),
                "final_reward": traj.get("final_reward", 0.0),
                "success": traj.get("success", False),
                "total_steps": traj.get("total_steps", 0),
                "steps": traj.get("steps", []),
                "metadata": {
                    "agent_model": traj.get("agent_model"),
                    "failure_class": traj.get("failure_class"),
                    "total_patch_ops": traj.get("total_patch_ops"),
                },
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("format", choices=["sft", "dpo", "rl", "all"])
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.output or REPO_ROOT / "data"
    if args.format in {"sft", "all"}:
        out = out_dir if args.format == "all" else (args.output or REPO_ROOT / "data" / "sft_pairs.jsonl")
        if args.format == "all":
            out = out_dir / "sft_pairs.jsonl"
        n = export_sft([args.input], out)
        print(f"[sft] {n} pairs -> {out}")
    if args.format in {"dpo", "all"}:
        out = (out_dir / "dpo_train.jsonl") if args.format == "all" else (args.output or REPO_ROOT / "data" / "dpo_train.jsonl")
        n = export_dpo(args.input, out)
        print(f"[dpo] {n} pairs -> {out}")
    if args.format in {"rl", "all"}:
        out = (out_dir / "rl_rollouts.jsonl") if args.format == "all" else (args.output or REPO_ROOT / "data" / "rl_rollouts.jsonl")
        n = export_rl(args.input, out)
        print(f"[rl] {n} rollouts -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
