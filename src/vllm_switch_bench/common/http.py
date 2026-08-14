from __future__ import annotations

import json
from typing import Any, Callable


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_explicit_ttft_s(obj: dict[str, Any]) -> float | None:
    for key in ("ttft_s", "ttft", "fttf_s", "fttf", "time_to_first_token_s", "time_to_first_token"):
        value = _to_float(obj.get(key))
        if value is not None:
            return value
    metrics = obj.get("metrics")
    if isinstance(metrics, dict):
        for key in (
            "ttft_s",
            "ttft",
            "fttf_s",
            "fttf",
            "time_to_first_token_s",
            "time_to_first_token",
        ):
            value = _to_float(metrics.get(key))
            if value is not None:
                return value
    return None


def parse_openai_stream_response(
    response,
    started_at: float,
    now_fn: Callable[[], float],
    *,
    ttft_mode: str = "first_chunk",
) -> dict[str, Any]:
    first_chunk_s = None
    explicit_ttft_s = None
    completed_at = None
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
            try:
                completed_at = now_fn()
            except Exception:
                completed_at = None
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
        if explicit_ttft_s is None:
            explicit_ttft_s = _extract_explicit_ttft_s(obj)
    output_text = "".join(text_parts)
    if ttft_mode == "first_chunk":
        ttft_s = first_chunk_s
    elif ttft_mode == "explicit":
        ttft_s = explicit_ttft_s
    else:
        raise ValueError(f"unsupported ttft_mode: {ttft_mode}")
    ttft_available = ttft_s is not None
    return {
        "ttft_s": ttft_s,
        "ttft_available": ttft_available,
        "output_text": output_text,
        "completion_tokens": completion_tokens,
        "completed_at": completed_at,
        "chunks": chunks,
    }
