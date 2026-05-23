"""Smoke test for the DeepSeek client.

Usage:
    python scripts/smoke_test_deepseek.py

Reads DEEPSEEK_API_KEY from process env or .env in repo root.
Returns non-zero if the API call fails or the response is empty.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from triton_change.llm_clients import DeepSeekClient


def main() -> int:
    try:
        client = DeepSeekClient()
    except ValueError as e:
        print(f"[smoke] config error: {e}")
        return 2

    print(f"[smoke] model={client.model}  base_url={client.base_url}")
    try:
        resp = client.chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise assistant. Reply with exactly one short word.",
                },
                {"role": "user", "content": "Say 'pong' and nothing else."},
            ],
            temperature=0.0,
            max_tokens=10,
        )
    except RuntimeError as e:
        print(f"[smoke] FAILED: {e}")
        return 1

    print(f"[smoke] response: {resp['content']!r}")
    print(f"[smoke] usage:    {resp['usage']}")
    print(f"[smoke] elapsed:  {resp['elapsed_s']:.2f}s")

    if not resp["content"].strip():
        print("[smoke] FAILED: empty response")
        return 1

    if "pong" in resp["content"].lower():
        print("[OK] DeepSeek client works.")
    else:
        print("[WARN] response did not contain 'pong'; client works but check prompt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
