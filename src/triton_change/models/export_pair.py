from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from triton_change.models.hybrid_model import HybridConfig, base_config, fake_batch, make_model, modified_config
from triton_change.triton.repository import create_python_backend_model


def export_onnx(config: HybridConfig, out: Path, name: str) -> Path:
    model = make_model(config)
    input_ids, _ = fake_batch(config, batch_size=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (input_ids,),
        out,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch"}, "logits": {0: "batch"}},
    )
    meta = {
        "name": name,
        "seq_len": config.seq_len,
        "embed_dim": config.embed_dim,
        "conv_channels": config.conv_channels,
        "transformer_ff": config.transformer_ff,
        "insert_precision_cast": config.insert_precision_cast,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate baseline and modified ONNX graphs plus a Triton repo.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--model-name", default="hybrid_text_model")
    args = parser.parse_args()

    onnx_dir = args.out_dir / "onnx"
    base_path = export_onnx(base_config(), onnx_dir / "hybrid_base.onnx", "hybrid_base")
    modified_path = export_onnx(modified_config(), onnx_dir / "hybrid_modified.onnx", "hybrid_modified")
    triton_dir = create_python_backend_model(
        model_dir=args.out_dir / "triton_repo" / args.model_name,
        model_name=args.model_name,
        onnx_path=base_path,
        input_seq_len=base_config().seq_len,
        num_classes=base_config().num_classes,
    )
    print(json.dumps({"base": str(base_path), "modified": str(modified_path), "triton_model_dir": str(triton_dir)}, indent=2))


if __name__ == "__main__":
    main()
