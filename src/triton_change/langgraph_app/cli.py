from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from triton_change.langgraph_app.logging import utc_stamp


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
    parser.add_argument("--log-dir", type=Path, default=Path("log"), help="Directory for node/tool/LLM JSON snapshots.")
    parser.add_argument("--run-id", default=None, help="Stable run id for logs. Defaults to a UTC timestamp.")
    args = parser.parse_args()

    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url
    if args.model:
        os.environ["OPENAI_MODEL"] = args.model

    from triton_change.langgraph_app.graph import build_graph

    app = build_graph()
    run_id = args.run_id or utc_stamp()
    print(f"[langgraph] run_id={run_id}", flush=True)
    print(f"[langgraph] log_dir={args.log_dir / run_id}", flush=True)
    result = app.invoke(
        {
            "base_onnx": str(args.base_onnx),
            "target_onnx": str(args.target_onnx),
            "triton_model_dir": str(args.triton_model_dir),
            "out_dir": str(args.out),
            "model_name": args.model_name or args.triton_model_dir.name,
            "log_dir": str(args.log_dir),
            "run_id": run_id,
        }
    )
    print(json.dumps(result.get("result", result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
