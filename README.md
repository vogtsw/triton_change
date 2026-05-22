# Triton Change

This project demonstrates two independent ways to analyze differences between two ONNX graphs and apply incremental changes to a Triton Python-backend model repository:

- `skills/triton-onnx-delta`: a Codex skill with small deterministic scripts.
- `src/triton_change/langgraph_app`: a LangGraph workflow using an OpenAI-compatible client.

The sample model combines `Conv1d` and `TransformerEncoder` layers. The export script creates a baseline ONNX graph and a modified graph with a longer input sequence, wider Conv1d branch, and an inserted precision cast.

## Quick Start

```powershell
py -m pip install -e ".[torch,langgraph]"
py -m triton_change.models.export_pair --out-dir artifacts
py -m triton_change.onnx_delta.cli artifacts\onnx\hybrid_base.onnx artifacts\onnx\hybrid_modified.onnx --out artifacts\delta.json
py -m triton_change.triton.patch_cli artifacts\triton_repo\hybrid_text_model artifacts\onnx\hybrid_base.onnx artifacts\onnx\hybrid_modified.onnx --out artifacts\triton_repo_patched
```

For DeepSeek or any OpenAI-compatible endpoint, configure environment variables instead of hardcoding secrets:

```powershell
$env:OPENAI_BASE_URL="https://api.deepseek.com"
$env:OPENAI_API_KEY="..."
$env:OPENAI_MODEL="deepseek-chat"
py -m triton_change.langgraph_app.cli artifacts\onnx\hybrid_base.onnx artifacts\onnx\hybrid_modified.onnx --triton-model-dir artifacts\triton_repo\hybrid_text_model --out artifacts\langgraph_patch
```

