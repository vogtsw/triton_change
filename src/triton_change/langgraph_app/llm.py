from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI


def complete_patch_review(delta: dict[str, Any], patch_plan: dict[str, Any], model: str | None = None) -> str:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return "LLM review skipped: OPENAI_API_KEY or DEEPSEEK_API_KEY is not set."
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = OpenAI(api_key=api_key, base_url=base_url)
    payload = {
        "delta_changes": delta.get("changes", [])[:40],
        "patch_plan": patch_plan,
    }
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You review ONNX graph diffs and Triton repository patch plans. Be concise and focus on serving risks.",
            },
            {
                "role": "user",
                "content": "Review this ONNX-to-Triton incremental patch plan and list any missing changes:\n"
                + json.dumps(payload, ensure_ascii=False, indent=2),
            },
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content or ""

