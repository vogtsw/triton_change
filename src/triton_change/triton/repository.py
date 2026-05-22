from __future__ import annotations

import shutil
from pathlib import Path


def model_py(input_seq_len: int, num_classes: int) -> str:
    return f'''from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import triton_python_backend_utils as pb_utils


MODEL_FILENAME = "model.onnx"
INPUT_TENSOR_NAME = "input_ids"
OUTPUT_TENSOR_NAME = "logits"
EXPECTED_SEQUENCE_LENGTH = {input_seq_len}
EXPECTED_NUM_CLASSES = {num_classes}
INPUT_DTYPE = np.int64
OUTPUT_DTYPE = np.float32
ORT_PROVIDERS = ["CPUExecutionProvider"]


class TritonPythonModel:
    """NVIDIA Triton Python backend entrypoint for the exported ONNX model."""

    def initialize(self, args):
        model_dir = Path(args["model_repository"]) / args["model_name"]
        model_path = model_dir / MODEL_FILENAME
        self.session = ort.InferenceSession(str(model_path), providers=ORT_PROVIDERS)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        if self.input_name != INPUT_TENSOR_NAME:
            raise pb_utils.TritonModelException(
                f"Expected ONNX input {{INPUT_TENSOR_NAME}}, got {{self.input_name}}"
            )

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, INPUT_TENSOR_NAME)
            input_ids = input_tensor.as_numpy().astype(INPUT_DTYPE, copy=False)
            if input_ids.ndim != 2 or input_ids.shape[1] != EXPECTED_SEQUENCE_LENGTH:
                raise pb_utils.TritonModelException(
                    f"Expected {{INPUT_TENSOR_NAME}} shape [batch, {{EXPECTED_SEQUENCE_LENGTH}}], "
                    f"got {{list(input_ids.shape)}}"
                )
            logits = self.session.run([self.output_name], {{self.input_name: input_ids}})[0]
            if logits.shape[-1] != EXPECTED_NUM_CLASSES:
                raise pb_utils.TritonModelException(
                    f"Expected {{OUTPUT_TENSOR_NAME}} last dim {{EXPECTED_NUM_CLASSES}}, got {{logits.shape[-1]}}"
                )
            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor(OUTPUT_TENSOR_NAME, logits.astype(OUTPUT_DTYPE, copy=False))
            ]))
        return responses
'''


def config_pbtxt(model_name: str, input_seq_len: int, num_classes: int) -> str:
    return f'''name: "{model_name}"
backend: "python"
max_batch_size: 0
input [
  {{
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1, {input_seq_len} ]
  }}
]
output [
  {{
    name: "logits"
    data_type: TYPE_FP32
    dims: [ -1, {num_classes} ]
  }}
]
instance_group [
  {{
    kind: KIND_CPU
  }}
]
'''


def create_python_backend_model(
    model_dir: Path,
    model_name: str,
    onnx_path: Path,
    input_seq_len: int,
    num_classes: int,
) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    version_dir = model_dir / "1"
    version_dir.mkdir(exist_ok=True)
    (version_dir / "model.py").write_text(model_py(input_seq_len, num_classes), encoding="utf-8")
    (model_dir / "config.pbtxt").write_text(config_pbtxt(model_name, input_seq_len, num_classes), encoding="utf-8")
    shutil.copy2(onnx_path, model_dir / "model.onnx")
    return model_dir
