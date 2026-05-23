"""End-to-end Phase 1 driver.

Pipeline (per spec section 5 + Phase 1 acceptance):

    oracle/patch_ops.json
      -> apply_patch_ops(old_model_triton.py)
         -> sandbox/candidate_model_triton.py
      -> static_check
      -> sandboxed correctness_check
      -> compute_reward
      -> tasks/<id>/reward.json + tasks/<id>/sandbox/run_log.json

Usage:
    python scripts/run_phase1.py tasks/task_000001
    python scripts/run_phase1.py tasks/task_000001 --device cuda
    python scripts/run_phase1.py tasks/task_000001 --skip-correctness
    python scripts/run_phase1.py tasks/task_000001 --patch-ops some_other.json

Exit codes:
    0  success (numerical correctness passed)
    1  any failure (static or correctness)
    2  configuration error (file missing, etc.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

from triton_change.correctness import correctness_check
from triton_change.patcher import apply_patch_ops
from triton_change.reward import compute_reward
from triton_change.static_check import static_check


def _load_json(p: Path):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task_dir", type=Path)
    ap.add_argument("--patch-ops", type=Path, default=None,
                    help="Path to patch_ops.json. Defaults to <task>/oracle/patch_ops.json.")
    ap.add_argument("--device", default="cpu", help="cpu or cuda[:N]")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--skip-correctness", action="store_true",
                    help="Run patch + static only (useful when Triton/CUDA unavailable).")
    args = ap.parse_args()

    task_dir: Path = args.task_dir.resolve()
    if not task_dir.is_dir():
        print(f"[err] task dir not found: {task_dir}")
        return 2

    patch_ops_path = args.patch_ops or (task_dir / "oracle" / "patch_ops.json")
    if not patch_ops_path.exists():
        print(f"[err] patch_ops file not found: {patch_ops_path}")
        return 2

    old_path = task_dir / "old_model_triton.py"
    if not old_path.exists():
        print(f"[err] {old_path} missing")
        return 2

    print(f"[phase1] task: {task_dir.name}")
    print(f"[phase1] patch ops: {patch_ops_path}")
    print(f"[phase1] device: {args.device}")

    # ---- 1. Patch ----
    # Candidate file is written at task root. The correctness check creates
    # its own (wipeable) `sandbox/` subdir to stage the candidate + data.
    src_text = old_path.read_text(encoding="utf-8")
    patch_doc = _load_json(patch_ops_path)
    ops = patch_doc.get("ops", [])

    patch_result = apply_patch_ops(
        src_text, ops, workspace=task_dir, candidate_filename="candidate_model_triton.py"
    )

    print(f"[phase1] patch: {len(ops)} ops, {'ok' if patch_result.all_succeeded else 'FAIL'}")
    if not patch_result.all_succeeded:
        for r in patch_result.op_results:
            mark = "ok" if r.success else "FAIL"
            print(f"  [{mark}] {r.op.get('operation')}: {r.detail}")
        rb = compute_reward(None, None, ops, [], patch_apply_error=True)
        _write_outputs(task_dir, patch_result, None, None, rb)
        return 1

    # ---- 2. Static check ----
    candidate_path = patch_result.candidate_path
    static_result = static_check(candidate_path)
    print(f"[phase1] static: {'pass' if static_result.passed else 'FAIL'}"
          + (f"  bad_imports={static_result.bad_imports}" if static_result.bad_imports else "")
          + (f"  danger={static_result.danger_findings}" if static_result.danger_findings else ""))
    if not static_result.passed:
        rb = compute_reward(static_result.to_dict(), None, ops, [])
        _write_outputs(task_dir, patch_result, static_result.to_dict(), None, rb)
        return 1

    # ---- 3. Correctness check ----
    correctness_dict = None
    if args.skip_correctness:
        print("[phase1] correctness: SKIPPED (--skip-correctness)")
    else:
        cr = correctness_check(candidate_path, task_dir, timeout=args.timeout, device=args.device)
        correctness_dict = cr.to_dict()
        verdict = "pass" if cr.passed else f"FAIL ({cr.failure_class})"
        print(f"[phase1] correctness: {verdict}  elapsed={cr.elapsed_s:.2f}s"
              + (f"  max_abs={cr.max_abs_error:.2e}" if cr.max_abs_error is not None else ""))
        if cr.error:
            print(f"  error: {cr.error[:300]}")

    # ---- 4. Reward ----
    semantic_labels_path = task_dir / "hidden_eval" / "semantic_change_labels.json"
    semantic_labels = []
    if semantic_labels_path.exists():
        semantic_labels = _load_json(semantic_labels_path).get("labels", [])

    rb = compute_reward(
        static_result=static_result.to_dict(),
        correctness_result=correctness_dict,
        patch_ops=ops,
        semantic_labels=semantic_labels,
    )

    print(f"[phase1] reward: total={rb.total:.4f}  success={rb.success}"
          + (f"  failure_class={rb.failure_class}" if rb.failure_class else ""))
    print("[phase1] reward components:")
    for k, v in rb.components.items():
        print(f"    {k:<32s} {v:+.2f}")

    _write_outputs(task_dir, patch_result, static_result.to_dict(), correctness_dict, rb)

    return 0 if rb.success else 1


def _write_outputs(
    task_dir: Path,
    patch_result,
    static_dict,
    correctness_dict,
    reward,
):
    sandbox = task_dir / "sandbox"
    sandbox.mkdir(exist_ok=True)

    log = {
        "task_id": task_dir.name,
        "patch_result": patch_result.to_dict(),
        "static_check": static_dict,
        "correctness_check": correctness_dict,
        "reward": reward.to_dict(),
    }
    (sandbox / "run_log.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (task_dir / "reward.json").write_text(
        json.dumps(reward.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    sys.exit(main())
