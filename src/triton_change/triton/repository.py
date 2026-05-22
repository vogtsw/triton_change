from __future__ import annotations

import shutil
from pathlib import Path


MODEL_PY = '''from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        model_dir = Path(args["model_repository"]) / args["model_name"]
        model_path = model_dir / "model.onnx"
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "input_ids")
            input_ids = input_tensor.as_numpy().astype(np.int64, copy=False)
            logits = self.session.run([self.output_name], {self.input_name: input_ids})[0]
            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("logits", logits.astype(np.float32, copy=False))
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
    (version_dir / "model.py").write_text(MODEL_PY, encoding="utf-8")
    (model_dir / "config.pbtxt").write_text(config_pbtxt(model_name, input_seq_len, num_classes), encoding="utf-8")
    shutil.copy2(onnx_path, model_dir / "model.onnx")
    return model_dir

