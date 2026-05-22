from triton_change.onnx_delta.schema import TensorInfo
from triton_change.triton.patcher import _triton_dims


def test_triton_dims_convert_symbolic_to_negative_one():
    info = TensorInfo(name="x", dtype="INT64", shape=["batch", 40])
    assert _triton_dims(info) == [-1, 40]

