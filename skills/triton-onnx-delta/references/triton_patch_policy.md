# Triton Patch Policy

Patch in this order:

1. Replace the executable ONNX graph.
2. Regenerate Triton input/output metadata from the target graph.
3. Preserve existing Python backend code when it only delegates to ONNX Runtime.
4. Patch backend code when it hardcodes tensor names, rank, dtype conversion, class count, sequence length, or output allocation.
5. Write a delta report beside the patched model.

Risk signals:

- Input shape or dtype changed: client payload contract changed.
- Output shape or dtype changed: downstream consumers may break.
- Cast count changed: precision behavior changed.
- Conv initializer rank 3 changed: Conv1d channel or kernel changed.
- Transformer attention operators changed: inspect sequence length, mask handling, and projection dimensions.

