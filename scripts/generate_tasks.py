"""Parameterized task generator (Phase 3a: 20 handcrafted + Phase 3b: 100 total).

Usage:
    python scripts/generate_tasks.py                       # all 100 tasks
    python scripts/generate_tasks.py --only task_000050
    python scripts/generate_tasks.py --from 21 --to 100    # Phase 3b batch only
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = REPO_ROOT / "tasks"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from task_config import TaskConfig  # noqa: E402
from task_profiles import render_task  # noqa: E402
from task_catalog import build_all_configs, tier_counts  # noqa: E402

TASK_CONFIGS: list[TaskConfig] = build_all_configs()

_SKIP_NN_VERIFY = {
    "gelu_bias", "gelu_relu", "rmsnorm", "dtype", "combo_hidden_dtype",
    "combo_shape_eps", "combo_shape_gelu_relu", "combo_rmsnorm_dtype",
}


# ---------- PyTorch model + ONNX export ----------


class FFNBlock(nn.Module):
    def __init__(self, hidden: int, intermediate: int, eps: float):
        super().__init__()
        self.ln = nn.LayerNorm(hidden, eps=eps)
        self.fc1 = nn.Linear(hidden, intermediate)
        self.fc2 = nn.Linear(intermediate, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.fc1(h)
        h = nn.functional.gelu(h, approximate="tanh")
        h = self.fc2(h)
        return h


def _export_onnx(
    model: nn.Module,
    hidden: int,
    seq_len: int,
    batch_size: int,
    out_path: Path,
    *,
    dynamic_batch: bool = False,
) -> None:
    model.eval()
    dummy = torch.randn(batch_size, seq_len, hidden)
    kwargs: dict[str, Any] = dict(
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
    )
    if dynamic_batch:
        kwargs["dynamic_axes"] = {"input": {0: "batch"}, "output": {0: "batch"}}
    with torch.no_grad():
        try:
            torch.onnx.export(model, dummy, str(out_path), dynamo=False, **kwargs)
        except TypeError:
            torch.onnx.export(model, dummy, str(out_path), **kwargs)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_meta(cfg: TaskConfig, expected_oracle_patch_ops: int) -> dict[str, Any]:
    return {
        "task_id": cfg.task_id,
        "tier": cfg.tier,
        "change_types": cfg.change_types,
        "base_model": "ln_linear_gelu_linear",
        "from": cfg.from_params,
        "to": cfg.to_params,
        "dtype": cfg.dtype_meta,
        "device": "cuda",
        "min_gpu_capability": "sm_70",
        "estimated_difficulty": cfg.estimated_difficulty,
        "expected_oracle_patch_ops": expected_oracle_patch_ops,
        "description": cfg.description,
    }


def _semantic_labels(cfg: TaskConfig) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    if cfg.hidden_old != cfg.hidden_new or cfg.inter_old != cfg.inter_new:
        labels.append({
            "label": "shape_param_change",
            "from": {"hidden_size": cfg.hidden_old, "intermediate_size": cfg.inter_old},
            "to": {"hidden_size": cfg.hidden_new, "intermediate_size": cfg.inter_new},
            "affected_region_hint": ",".join(filter(None, [
                "constant:HIDDEN_SIZE" if cfg.hidden_old != cfg.hidden_new else "",
                "constant:INTERMEDIATE_SIZE" if cfg.inter_old != cfg.inter_new else "",
            ])),
            "confidence": 1.0,
        })
    if cfg.seq_old != cfg.seq_new:
        labels.append({
            "label": "shape_param_change",
            "from": {"seq_len": cfg.seq_old},
            "to": {"seq_len": cfg.seq_new},
            "affected_region_hint": "constant:SEQ_LEN",
            "confidence": 1.0,
        })
    if cfg.profile == "batch":
        labels.append({
            "label": "shape_param_change",
            "from": {"batch": "fixed", "fixed_batch": cfg.batch_old},
            "to": {"batch": "dynamic"},
            "affected_region_hint": "function:model_forward",
            "confidence": 1.0,
        })
    if cfg.eps_old != cfg.eps_new:
        labels.append({
            "label": "norm_change",
            "from": {"ln_eps": cfg.eps_old},
            "to": {"ln_eps": cfg.eps_new},
            "affected_region_hint": "constant:LN_EPS",
            "confidence": 1.0,
        })
    if cfg.from_params.get("dtype_name") != cfg.to_params.get("dtype_name"):
        labels.append({
            "label": "dtype_change",
            "from": {"dtype": cfg.from_params.get("dtype_name", "float32")},
            "to": {"dtype": cfg.to_params.get("dtype_name", "float32")},
            "affected_region_hint": "constant:DTYPE_NAME",
            "confidence": 1.0,
        })
    if cfg.profile in {"gelu_bias", "gelu_relu"}:
        labels.append({
            "label": "activation_change",
            "from": {"activation": "GELU"},
            "to": {"activation": "BiasGELU" if cfg.profile == "gelu_bias" else "ReLU"},
            "affected_region_hint": "kernel:gelu_act_kernel",
            "confidence": 1.0,
        })
    if cfg.profile == "rmsnorm":
        labels.append({
            "label": "norm_change",
            "from": {"norm": "LayerNorm"},
            "to": {"norm": "RMSNorm"},
            "affected_region_hint": "kernel:layernorm_fwd_kernel",
            "confidence": 1.0,
        })
    return labels


def _build_diff_summary(cfg: TaskConfig, ops: list[dict[str, Any]]) -> dict[str, Any]:
    input_shape_old = [cfg.batch_old, cfg.seq_old, cfg.hidden_old]
    input_shape_new = [cfg.batch_new, cfg.seq_new, cfg.hidden_new]
    return {
        "raw_diff": {
            "input_shape_changes": (
                [{"name": "input", "old_shape": input_shape_old, "new_shape": input_shape_new}]
                if input_shape_old != input_shape_new else []
            ),
            "output_shape_changes": (
                [{"name": "output", "old_shape": input_shape_old, "new_shape": input_shape_new}]
                if input_shape_old != input_shape_new else []
            ),
            "input_dtype_changes": (
                [{"name": "input", "old_dtype": cfg.from_params.get("dtype_name", "float32"),
                  "new_dtype": cfg.to_params.get("dtype_name", "float32")}]
                if cfg.from_params.get("dtype_name") != cfg.to_params.get("dtype_name") else []
            ),
            "output_dtype_changes": [],
            "op_count_delta": {},
            "weight_shape_changes": _weight_shape_changes(cfg),
            "attribute_changes": (
                [{"op_type": "LayerNormalization", "field": "epsilon", "old": cfg.eps_old, "new": cfg.eps_new}]
                if cfg.eps_old != cfg.eps_new else []
            ),
            "activation_changes": (
                [{"from": "GELU", "to": "BiasGELU" if cfg.profile == "gelu_bias" else "ReLU"}]
                if cfg.profile in {"gelu_bias", "gelu_relu"} else []
            ),
            "norm_changes": (
                [{"from": "LayerNorm", "to": "RMSNorm"}] if cfg.profile == "rmsnorm" else []
            ),
            "batch_changes": (
                [{"from": f"fixed_{cfg.batch_old}", "to": "dynamic"}] if cfg.profile == "batch" else []
            ),
        },
        "semantic_labels": _semantic_labels(cfg),
        "summary_text": cfg.description,
    }


def _weight_shape_changes(cfg: TaskConfig) -> list[dict[str, Any]]:
    if cfg.hidden_old == cfg.hidden_new and cfg.inter_old == cfg.inter_new:
        return []
    return [
        {"name": "ln.weight", "old_shape": [cfg.hidden_old], "new_shape": [cfg.hidden_new]},
        {"name": "ln.bias", "old_shape": [cfg.hidden_old], "new_shape": [cfg.hidden_new]},
        {"name": "fc1.weight", "old_shape": [cfg.inter_old, cfg.hidden_old], "new_shape": [cfg.inter_new, cfg.hidden_new]},
        {"name": "fc1.bias", "old_shape": [cfg.inter_old], "new_shape": [cfg.inter_new]},
        {"name": "fc2.weight", "old_shape": [cfg.hidden_old, cfg.inter_old], "new_shape": [cfg.hidden_new, cfg.inter_new]},
        {"name": "fc2.bias", "old_shape": [cfg.hidden_old], "new_shape": [cfg.hidden_new]},
    ]


def _build_input_specs(cfg: TaskConfig) -> dict[str, Any]:
    tol = {"atol": 1e-3, "rtol": 1e-3} if cfg.dtype_meta in {"fp16", "bf16"} else {"atol": 1e-4, "rtol": 1e-4}
    return {
        "input": {
            "name": "input",
            "shape": [cfg.batch_new, cfg.seq_new, cfg.hidden_new],
            "dtype": "fp32",
        },
        "output": {
            "name": "output",
            "shape": [cfg.batch_new, cfg.seq_new, cfg.hidden_new],
            "dtype": "fp32",
        },
        "weight_keys": ["ln_w", "ln_b", "w1", "b1", "w2", "b2"],
        "weight_shapes": {
            "ln_w": [cfg.hidden_new],
            "ln_b": [cfg.hidden_new],
            "w1": [cfg.inter_new, cfg.hidden_new],
            "b1": [cfg.inter_new],
            "w2": [cfg.hidden_new, cfg.inter_new],
            "b2": [cfg.hidden_new],
        },
        "test_input_count": 3,
        "tolerance": tol,
    }


def generate_one(cfg: TaskConfig) -> None:
    print(f"\n=== {cfg.task_id} ({cfg.profile}) ===")
    task_dir = TASKS_DIR / cfg.task_id
    (task_dir / "oracle").mkdir(parents=True, exist_ok=True)
    (task_dir / "hidden_eval").mkdir(parents=True, exist_ok=True)

    rendered = render_task(cfg)
    ops = rendered.patch_ops
    assert len(ops) <= 8, f"{cfg.task_id}: oracle has {len(ops)} ops (>8 spec limit)"

    (task_dir / "old_model_triton.py").write_text(rendered.old_triton, encoding="utf-8")
    (task_dir / "oracle" / "new_model_triton.py").write_text(rendered.new_triton, encoding="utf-8")
    (task_dir / "oracle" / "cpu_demo_new_model_triton.py").write_text(rendered.cpu_demo, encoding="utf-8")
    (task_dir / "hidden_eval" / "reference_forward.py").write_text(rendered.reference, encoding="utf-8")

    meta = _build_meta(cfg, expected_oracle_patch_ops=len(ops))
    (task_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    patch_doc = {"task_id": cfg.task_id, "ops": ops, "notes": cfg.description}
    (task_dir / "oracle" / "patch_ops.json").write_text(
        json.dumps(patch_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    diff = _build_diff_summary(cfg, ops)
    (task_dir / "oracle" / "diff_summary.json").write_text(
        json.dumps(diff, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    specs = _build_input_specs(cfg)
    (task_dir / "hidden_eval" / "input_specs.json").write_text(
        json.dumps(specs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    sem_labels = {"labels": _semantic_labels(cfg)}
    (task_dir / "hidden_eval" / "semantic_change_labels.json").write_text(
        json.dumps(sem_labels, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    cpu_demo_doc = {
        "task_id": cfg.task_id,
        "ops": [{
            "operation": "full_file_replace",
            "path": "candidate_model_triton.py",
            "new_code": rendered.cpu_demo,
            "reason": "CPU demo (no Triton): torch-only equivalent for Windows pipeline verification.",
        }],
        "notes": "Demo path; real oracle is oracle/patch_ops.json.",
    }
    (task_dir / "oracle" / "cpu_demo_patch_ops.json").write_text(
        json.dumps(cpu_demo_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"  [text] meta + 3 .py + 5 .json ({len(ops)} patch ops)")

    # Dynamic ONNX / tensors
    print("  [pyt]  generating ONNX/weights/inputs/outputs ...")
    base_seed = 42 + cfg.seed_offset
    target_seed = base_seed + 1
    inputs_seed = base_seed + 100

    torch.manual_seed(base_seed)
    base_model = FFNBlock(cfg.hidden_old, cfg.inter_old, eps=cfg.eps_old)
    _export_onnx(
        base_model, cfg.hidden_old, cfg.seq_old, cfg.batch_old,
        task_dir / "base.onnx", dynamic_batch=False,
    )

    torch.manual_seed(target_seed)
    target_model = FFNBlock(cfg.hidden_new, cfg.inter_new, eps=cfg.eps_new)
    _export_onnx(
        target_model, cfg.hidden_new, cfg.seq_new, cfg.batch_new,
        task_dir / "target.onnx", dynamic_batch=cfg.dynamic_batch_target,
    )

    weights = {
        "ln_w": target_model.ln.weight.detach().clone(),
        "ln_b": target_model.ln.bias.detach().clone(),
        "w1": target_model.fc1.weight.detach().clone(),
        "b1": target_model.fc1.bias.detach().clone(),
        "w2": target_model.fc2.weight.detach().clone(),
        "b2": target_model.fc2.bias.detach().clone(),
    }
    torch.save(weights, task_dir / "hidden_eval" / "weights.pt")

    torch.manual_seed(inputs_seed)
    test_inputs = [torch.randn(cfg.batch_new, cfg.seq_new, cfg.hidden_new) for _ in range(3)]
    torch.save(test_inputs, task_dir / "hidden_eval" / "test_inputs.pt")

    ref_mod = _load_module(task_dir / "hidden_eval" / "reference_forward.py", f"{cfg.task_id}_ref")
    target_outputs = []
    target_model.eval()
    with torch.no_grad():
        for x in test_inputs:
            y_ref = ref_mod.reference_forward(
                x, weights["ln_w"], weights["ln_b"],
                weights["w1"], weights["b1"], weights["w2"], weights["b2"],
            )
            target_outputs.append(y_ref)
            if cfg.profile not in _SKIP_NN_VERIFY:
                y_nn = target_model(x)
                err = (y_ref - y_nn).abs().max().item()
                if err > 1e-4:
                    raise RuntimeError(f"{cfg.task_id}: reference diverges from nn.Module (max err {err})")
    torch.save(target_outputs, task_dir / "hidden_eval" / "target_outputs.pt")

    print(f"  [ok]   {cfg.task_id} written")


def _task_num(task_id: str) -> int:
    return int(task_id.split("_", 1)[1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None, help="Generate only this task_id")
    ap.add_argument("--from", dest="from_n", type=int, default=None, metavar="N",
                    help="Generate tasks with index >= N (e.g. 21)")
    ap.add_argument("--to", dest="to_n", type=int, default=None, metavar="N",
                    help="Generate tasks with index <= N (e.g. 100)")
    args = ap.parse_args()

    targets = TASK_CONFIGS
    if args.only:
        targets = [c for c in TASK_CONFIGS if c.task_id == args.only]
        if not targets:
            print(f"[err] unknown task_id: {args.only}")
            return 1
    else:
        if args.from_n is not None:
            targets = [c for c in targets if _task_num(c.task_id) >= args.from_n]
        if args.to_n is not None:
            targets = [c for c in targets if _task_num(c.task_id) <= args.to_n]

    counts = tier_counts()
    print(f"[info] catalog: {len(TASK_CONFIGS)} tasks — {counts}")

    for cfg in targets:
        generate_one(cfg)

    print(f"\n[done] {len(targets)} task(s) generated (total catalog: {len(TASK_CONFIGS)}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
