---
name: triton-onnx-delta
description: Analyze architecture differences between two ONNX models and incrementally patch a Triton Python-backend model repository. Use when comparing baseline vs modified ONNX graphs, identifying Conv1d/Transformer/precision/shape changes, or updating Triton config/model artifacts from graph deltas.
---

# Triton ONNX Delta

Use this skill to compare two ONNX graphs and update an existing Triton model repository with the smallest safe serving changes.

## Workflow

1. Run deterministic ONNX diff:
   ```powershell
   py skills/triton-onnx-delta/scripts/analyze_onnx_delta.py BASE.onnx TARGET.onnx --out delta.json
   ```
2. Inspect `changes` for serving-contract updates:
   - `input` or `output`: patch `config.pbtxt` dims/data type and client payload expectations.
   - `initializer`: model weights, channel width, hidden size, or precision changed.
   - `op_count: Cast`: precision-local behavior changed.
   - `op_count: Conv`: Conv/Conv1d branch changed.
   - `node_sequence`: graph topology changed; backend assumptions need review.
3. Apply Triton incremental patch:
   ```powershell
   py skills/triton-onnx-delta/scripts/patch_triton_repo.py TRITON_MODEL_DIR BASE.onnx TARGET.onnx --out patched_repo
   ```
4. Review generated files:
   - `model.onnx`: replaced with target graph.
   - `config.pbtxt`: regenerated from target graph inputs/outputs.
   - `delta_report.json`: audit trail.
   - `precision_notes.md`: created only when Cast count changed.

## Policy

- Do not download CUDA or assume GPU execution.
- Prefer CPU-compatible Triton Python backend patches unless the target repository explicitly uses another backend.
- Never hardcode API keys or provider credentials into generated files.
- Treat output shape/dtype changes as client contract changes.
- If backend Python code contains hardcoded tensor names, shapes, or dtype casts, patch those alongside `config.pbtxt`.

For detailed patch review heuristics, read `references/triton_patch_policy.md`.

