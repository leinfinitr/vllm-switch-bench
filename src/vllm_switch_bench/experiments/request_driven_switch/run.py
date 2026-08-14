from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any

import httpx

from vllm_switch_bench.common.traces import REQUIRED_FIELDS, load_manifest
from vllm_switch_bench.common.provenance import file_metadata, git_metadata, repository_root
from vllm_switch_bench.common.schema import PROMPTS


def parse_sse_events(buffer: bytes) -> tuple[list[bytes], bytes]:
    normalized = buffer.replace(b"\r\n", b"\n")
    parts = normalized.split(b"\n\n")
    if normalized.endswith(b"\n\n"):
        return [part for part in parts[:-1] if part], b""
    return [part for part in parts[:-1] if part], parts[-1]


def _semantic_text(event: bytes) -> tuple[str, int | None, bool]:
    data_lines: list[bytes] = []
    for line in event.splitlines():
        if line.startswith(b"data:"):
            data_lines.append(line[5:].lstrip())
    if not data_lines:
        return "", None, False
    data = b"\n".join(data_lines).strip()
    if data == b"[DONE]":
        return "", None, True
    try:
        obj = json.loads(data)
    except (TypeError, ValueError) as exc:
        raise ValueError("malformed SSE data event") from exc
    if obj.get("error"):
        error = obj["error"]
        message = error.get("message") if isinstance(error, dict) else str(error)
        raise ValueError(f"SSE error event: {message or error}")
    choice = (obj.get("choices") or [{}])[0]
    delta = choice.get("delta") or {}
    text = delta.get("content") or choice.get("text") or ""
    usage = obj.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    return str(text), int(completion_tokens) if completion_tokens is not None else None, False


async def _dispatch_one(
    client: httpx.AsyncClient,
    base_url: str,
    row: dict[str, Any],
    trace_started: float,
    timeout_s: float = 600,
) -> dict[str, Any]:
    scheduled = trace_started + float(row["scheduled_offset_s"])
    await asyncio.sleep(max(0.0, scheduled - time.monotonic()))
    dispatched = time.monotonic()
    record: dict[str, Any] = {
        **{field: row[field] for field in REQUIRED_FIELDS},
        "client_dispatch_offset_s": dispatched - trace_started,
        "dispatch_lag_ms": (dispatched - scheduled) * 1000,
        "status": None,
        "error": None,
        "response_body_first_byte_ms": None,
        "semantic_ttft_ms": None,
        "trace_semantic_ttft_ms": None,
        "completion_latency_ms": None,
        "completion_tokens": None,
        "tpot_ms": None,
        "output_text": "",
        "stream_done": False,
    }
    prompt = PROMPTS[str(row["prompt_name"])]
    body: dict[str, Any] = {
        "model": row["model"],
        "max_tokens": int(row.get("max_tokens", prompt["max_tokens"])),
        "temperature": row.get("temperature", 0),
        "seed": row.get("seed", 1),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    endpoint = str(row["endpoint"])
    if endpoint == "/v1/chat/completions":
        body["messages"] = [{"role": "user", "content": prompt["prompt"]}]
    else:
        body["prompt"] = prompt["prompt"]

    buffer = b""
    first_byte = None
    semantic_at = None
    output_parts: list[str] = []

    async def consume_stream() -> None:
        nonlocal buffer, first_byte, semantic_at
        async with client.stream("POST", f"{base_url}{endpoint}", json=body) as response:
            record["status"] = response.status_code
            content_type = response.headers.get("content-type", "").lower()
            if "text/event-stream" not in content_type:
                raise ValueError(f"unexpected content-type: {content_type}")
            async for chunk in response.aiter_bytes():
                now = time.monotonic()
                if chunk and first_byte is None:
                    first_byte = now
                    record["response_body_first_byte_ms"] = (now - dispatched) * 1000
                buffer += chunk
                events, buffer = parse_sse_events(buffer)
                for event in events:
                    text, tokens, done = _semantic_text(event)
                    if done:
                        record["stream_done"] = True
                    if text:
                        if semantic_at is None:
                            semantic_at = now
                            record["semantic_ttft_ms"] = (now - dispatched) * 1000
                            record["trace_semantic_ttft_ms"] = (now - scheduled) * 1000
                        output_parts.append(text)
                    if tokens is not None:
                        record["completion_tokens"] = tokens
            if buffer.strip():
                raise ValueError("incomplete SSE event at end of stream")

    try:
        async with asyncio.timeout(timeout_s):
            await consume_stream()
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        record["completion_latency_ms"] = (time.monotonic() - dispatched) * 1000
        record["output_text"] = "".join(output_parts)
        tokens = record["completion_tokens"]
        if tokens is not None and tokens >= 2 and semantic_at is not None:
            record["tpot_ms"] = (record["completion_latency_ms"] - record["semantic_ttft_ms"]) / (
                tokens - 1
            )
    return record


def failed_record(record: dict[str, Any]) -> bool:
    status = record.get("status")
    timing_fields = ("semantic_ttft_ms", "completion_latency_ms", "tpot_ms")
    timings_valid = all(
        value is None or (isinstance(value, (int, float)) and math.isfinite(float(value)))
        for value in (record.get(field) for field in timing_fields)
    )
    return (
        status is None
        or not 200 <= int(status) < 300
        or bool(record.get("error"))
        or not bool(record.get("stream_done"))
        or record.get("semantic_ttft_ms") is None
        or not str(record.get("output_text") or "").strip()
        or not timings_valid
    )


async def run_trace(
    client: httpx.AsyncClient,
    base_url: str,
    rows: list[dict[str, Any]],
    timeout_s: float = 600,
) -> list[dict[str, Any]]:
    trace_started = time.monotonic()
    tasks = [
        asyncio.create_task(
            _dispatch_one(client, base_url.rstrip("/"), row, trace_started, timeout_s)
        )
        for row in rows
    ]
    records = await asyncio.gather(*tasks)
    return sorted(records, key=lambda record: str(record["request_id"]))


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a frozen OpenAI request-switch trace")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--runtime-repo",
        action="append",
        default=[],
        type=Path,
        help="Runtime repository used by the service. Repeat for controller/router and engine.",
    )
    parser.add_argument(
        "--runtime-file",
        action="append",
        default=[],
        type=Path,
        help="Runtime config or executable used by the service. Repeat as needed.",
    )
    parser.add_argument("--timeout-s", type=float, default=600)
    args = parser.parse_args(argv)
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")
    return args


async def main_async(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_manifest(args.manifest)
    async with httpx.AsyncClient(timeout=args.timeout_s, trust_env=False) as client:
        records = await run_trace(client, args.base_url, rows, args.timeout_s)
    write_jsonl(args.output, records)
    metadata_path = Path(args.output).with_suffix(".run.json")
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_url": args.base_url,
                "manifest": file_metadata(Path(args.manifest)),
                "benchmark_repo": git_metadata(repository_root()),
                "runtime_repositories": [git_metadata(path) for path in args.runtime_repo],
                "runtime_files": [file_metadata(path) for path in args.runtime_file],
                "requests": len(records),
                "failed": sum(failed_record(record) for record in records),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    failed = sum(failed_record(record) for record in records)
    print(json.dumps({"requests": len(records), "failed": failed, "output": args.output}))
    if failed:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(main_async(argv))


if __name__ == "__main__":
    raise SystemExit(main())
