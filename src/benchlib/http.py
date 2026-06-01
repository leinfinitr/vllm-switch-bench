from __future__ import annotations

import json
from typing import Any, Callable


def parse_openai_stream_response(response, started_at: float, now_fn: Callable[[], float]) -> dict[str, Any]:
    first_chunk_s = None
    chunks: list[str] = []
    completion_tokens = None
    for raw in response.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        if first_chunk_s is None:
            first_chunk_s = now_fn() - started_at
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data.strip() == "[DONE]":
            break
        chunks.append(data)
    text_parts: list[str] = []
    for chunk in chunks:
        try:
            obj = json.loads(chunk)
        except Exception:
            continue
        choice = obj.get("choices", [{}])[0]
        delta = choice.get("delta") or {}
        if "content" in delta and delta["content"]:
            text_parts.append(delta["content"])
        if "text" in choice and choice["text"]:
            text_parts.append(choice["text"])
        usage = obj.get("usage")
        if usage:
            completion_tokens = usage.get("completion_tokens")
    output_text = "".join(text_parts)
    return {
        "ttft_s": first_chunk_s,
        "output_text": output_text,
        "completion_tokens": completion_tokens,
        "chunks": chunks,
    }
