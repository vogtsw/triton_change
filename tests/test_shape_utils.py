from triton_change.onnx_delta.schema import TensorInfo
from triton_change.triton.patcher import _triton_dims
from triton_change.triton.patch_ops import inspect_triton_model
from triton_change.triton.repository import model_py


def test_triton_dims_convert_symbolic_to_negative_one():
    info = TensorInfo(name="x", dtype="INT64", shape=["batch", 40])
    assert _triton_dims(info) == [-1, 40]


def test_inspect_triton_model_handles_missing_binary_content(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.pbtxt").write_text('name: "m"\n', encoding="utf-8")
    (model_dir / "model.onnx").write_bytes(b"onnx")
    context = inspect_triton_model(model_dir)
    assert context["files"]["config.pbtxt"]["kind"] == "text"
    assert context["files"]["model.onnx"]["kind"] == "binary"


def test_generated_triton_model_py_contains_serving_guards():
    text = model_py(input_seq_len=40, num_classes=5)
    assert "EXPECTED_SEQUENCE_LENGTH = 40" in text
    assert "EXPECTED_NUM_CLASSES = 5" in text
    assert "TritonModelException" in text
