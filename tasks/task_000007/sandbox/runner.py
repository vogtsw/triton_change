"""Auto-generated correctness runner. Do not edit."""
import importlib.util
import json
import sys
import traceback
from pathlib import Path

DEVICE = "cpu"
WEIGHT_KEYS = ['ln_w', 'ln_b', 'w1', 'b1', 'w2', 'b2']

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE / "candidate_model_triton.py"
WEIGHTS = HERE / "weights.pt"
INPUTS = HERE / "test_inputs.pt"
OUTPUTS = HERE / "candidate_outputs.pt"
ERR = HERE / "runner_error.json"


def _err(failure_class, message):
    ERR.write_text(
        json.dumps({"failure_class": failure_class, "error": message}, ensure_ascii=False)
    )
    return 2


try:
    import torch
except Exception as e:
    sys.exit(_err("import", f"failed to import torch: {e!r}"))

try:
    spec = importlib.util.spec_from_file_location("candidate_model_triton", CANDIDATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
except Exception as e:
    tb = traceback.format_exc(limit=20)
    sys.exit(_err("import", f"failed to load candidate: {e!r}\n{tb}"))

if not hasattr(mod, "model_forward"):
    sys.exit(_err("import", "candidate has no model_forward"))

try:
    weights = torch.load(WEIGHTS, weights_only=False)
    inputs = torch.load(INPUTS, weights_only=False)
except Exception as e:
    sys.exit(_err("runtime", f"failed to load weights/inputs: {e!r}"))

outputs = []
try:
    for x in inputs:
        x_d = x.to(DEVICE) if DEVICE != "cpu" else x
        wargs = {k: (v.to(DEVICE) if DEVICE != "cpu" else v) for k, v in weights.items()}
        y = mod.model_forward(x_d, *(wargs[k] for k in WEIGHT_KEYS))
        outputs.append(y.detach().to("cpu"))
except Exception as e:
    tb = traceback.format_exc(limit=20)
    sys.exit(_err("runtime", f"model_forward failed: {e!r}\n{tb}"))

torch.save(outputs, OUTPUTS)
sys.exit(0)
