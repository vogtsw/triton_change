# Improved Triton Mapping Rules

This file describes custom ONNX-to-Triton code mapping rules for an improved Python backend that is not a stock NVIDIA Triton example.

The LangGraph workflow should read this Markdown together with the ONNX diff JSON and the existing Triton code preview. The LLM must still output only compact patch operations. Local tools apply the actual file edits.

## Serving Contract Fields

Some improved Triton Python backends use custom contract constants instead of standard names:

- `MODEL_SEQUENCE_TOKENS`: target sequence length for `input_ids`.
- `MODEL_NUM_CLASSES`: target last dimension for `logits`.
- `MODEL_INPUT_DTYPE`: numpy dtype used for request input conversion.
- `MODEL_OUTPUT_DTYPE`: numpy dtype used for response tensor conversion.

## ONNX Diff to Code Rules

When the ONNX diff has an `input` change for `input_ids`:

- Read the target shape from `after.shape`.
- Use the final dimension as the sequence length.
- Patch `config.pbtxt` so the `input_ids` dims match the target ONNX shape.
- If `1/model.py` contains `MODEL_SEQUENCE_TOKENS = <int>`, patch that constant to the target sequence length.
- If `1/model.py` contains shape checks using `MODEL_SEQUENCE_TOKENS`, do not rewrite the whole check. Updating the constant is enough.

When the ONNX diff has an `output` change for `logits`:

- Read the target shape from `after.shape`.
- Use the final dimension as the class count.
- Patch `config.pbtxt` so the `logits` dims match the target ONNX shape.
- If `1/model.py` contains `MODEL_NUM_CLASSES = <int>`, patch that constant to the target class count.

When the ONNX diff shows only initializer/internal operator changes:

- Replace `model.onnx`.
- Copy any ONNX external data sidecar files.
- Do not patch Python preprocessing/postprocessing unless a serving contract field above is affected.

When `Cast` operator count changes:

- Do not change Python dtype conversion automatically unless the input/output tensor dtype in the ONNX serving contract changed.
- Record the change in the delta report and patch summary.

## Preferred Patch Operations

Use these operations:

- `copy_target_onnx` for replacing ONNX and external data.
- `regex_replace` for constants such as `MODEL_SEQUENCE_TOKENS = 32`.
- `regex_replace` for `config.pbtxt` dtype or dims.
- `write_delta_report` for writing the full diff report.

Do not emit complete source files.
