from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from triton_change.langgraph_app.logging import estimate_tokens


def make_patch_ops(
    compact_delta: dict[str, Any],
    triton_context: dict[str, Any],
    fallback_ops: list[dict[str, Any]],
    model: str | None = None,
) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        output = {"patch_ops": fallback_ops, "notes": "LLM skipped; deterministic fallback patch ops used."}
        return {
            "patch_ops": fallback_ops,
            "raw": json.dumps(output, ensure_ascii=False),
            "usage": {
                "name": "make_patch_ops",
                "provider": "deterministic_fallback",
                "input_tokens": estimate_tokens(json.dumps({"compact_delta": compact_delta, "triton_context": triton_context}, ensure_ascii=False)),
                "output_tokens": estimate_tokens(json.dumps(output, ensure_ascii=False)),
            },
        }

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = OpenAI(api_key=api_key, base_url=base_url)
    payload = {
        "compact_delta": compact_delta,
        "triton_context": triton_context,
        "allowed_operations": [
            "copy_target_onnx",
            "regex_replace",
            "replace_text",
            "write_delta_report",
        ],
        "fallback_ops": fallback_ops,
    }
    user_content = (
        "Generate minimal JSON patch operations to update the Triton model repository for the target ONNX graph. "
        "Do not output full files. Output only small operation objects with path, operation, selector/pattern, and replacement when needed. "
        "Inspect Triton Python backend previews for hardcoded serving assumptions such as EXPECTED_SEQUENCE_LENGTH, "
        "EXPECTED_NUM_CLASSES, tensor names, dtype casts, and shape validation. Patch those assumptions when the ONNX diff changes them. "
        "Use fallback_ops unchanged when they are sufficient; otherwise return the smallest corrected set. Return JSON with key patch_ops.\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You produce structured patch operations for a tool to apply. "
                    "Never emit complete source files or secrets. Keep replacements minimal."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"patch_ops": fallback_ops, "parse_error": raw}
    usage_obj = response.usage
    usage = {
        "name": "make_patch_ops",
        "provider": base_url,
        "model": model,
        "input_tokens": getattr(usage_obj, "prompt_tokens", 0) if usage_obj else 0,
        "output_tokens": getattr(usage_obj, "completion_tokens", 0) if usage_obj else 0,
        "total_tokens": getattr(usage_obj, "total_tokens", 0) if usage_obj else 0,
    }
    return {"patch_ops": parsed.get("patch_ops", fallback_ops), "raw": raw, "usage": usage}
