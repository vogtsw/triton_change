from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LangGraph ONNX delta to Triton patch workflow.")
    parser.add_argument("base_onnx", type=Path)
    parser.add_argument("target_onnx", type=Path)
    parser.add_argument("--triton-model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/langgraph_patch"))
    parser.add_argument("--model-name")
    parser.add_argument("--api-key", help="OpenAI-compatible API key. Prefer env vars for normal use.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL, for example https://api.deepseek.com")
    parser.add_argument("--model", help="OpenAI-compatible model name.")
    args = parser.parse_args()

    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url
    if args.model:
        os.environ["OPENAI_MODEL"] = args.model

    from triton_change.langgraph_app.graph import build_graph

    app = build_graph()
    result = app.invoke(
        {
            "base_onnx": str(args.base_onnx),
            "target_onnx": str(args.target_onnx),
            "triton_model_dir": str(args.triton_model_dir),
            "out_dir": str(args.out),
            "model_name": args.model_name or args.triton_model_dir.name,
        }
    )
    print(json.dumps(result.get("result", result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

