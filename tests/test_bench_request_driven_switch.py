import asyncio
import sys
from pathlib import Path

import httpx

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from httpx import ASGITransport

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench_request_driven_switch import failed_record, parse_sse_events, run_trace


def test_parse_sse_events_handles_multiple_events_in_one_raw_chunk():
    chunk = (
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    events, remainder = parse_sse_events(chunk)
    assert remainder == b""
    assert len(events) == 3
    assert b'"content":"hello"' in events[1]


def test_run_trace_dispatches_overlapping_requests_and_keeps_failures():
    asyncio.run(_run_overlapping_trace())


async def _run_overlapping_trace():
    app = FastAPI()
    first_started = asyncio.Event()
    allow_first = asyncio.Event()

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        async def stream():
            if body["model"] == "a":
                first_started.set()
                await allow_first.wait()
            yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    rows = [
        {
            "request_id": "r1",
            "scheduled_offset_s": 0.0,
            "model": "a",
            "endpoint": "/v1/chat/completions",
            "prompt_name": "short_short",
            "max_tokens": 8,
            "temperature": 0,
            "stream": True,
            "seed": 1,
        },
        {
            "request_id": "r2",
            "scheduled_offset_s": 0.01,
            "model": "b",
            "endpoint": "/v1/chat/completions",
            "prompt_name": "short_short",
            "max_tokens": 8,
            "temperature": 0,
            "stream": True,
            "seed": 1,
        },
    ]
    async with httpx.AsyncClient(
        transport=ASGITransport(app), base_url="http://server", timeout=2
    ) as client:
        task = asyncio.create_task(run_trace(client, "http://server", rows))
        await first_started.wait()
        await asyncio.sleep(0.03)
        allow_first.set()
        records = await task

    assert [record["request_id"] for record in records] == ["r1", "r2"]
    assert all(record["status"] == 200 for record in records)
    assert all(record["semantic_ttft_ms"] is not None for record in records)
    assert records[1]["completion_latency_ms"] < records[0]["completion_latency_ms"]


def test_run_trace_keeps_broken_stream_and_timeout_records():
    asyncio.run(_run_failed_trace())


async def _run_failed_trace():
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        async def stream():
            if body["model"] == "slow":
                await asyncio.sleep(0.05)
            yield b"not-sse"
            raise RuntimeError("broken")

        return StreamingResponse(stream(), media_type="text/event-stream")

    rows = [
        {
            "request_id": "r1",
            "scheduled_offset_s": 0.0,
            "model": "broken",
            "endpoint": "/v1/chat/completions",
            "prompt_name": "short_short",
            "max_tokens": 8,
            "temperature": 0,
            "stream": True,
            "seed": 1,
        },
        {
            "request_id": "r2",
            "scheduled_offset_s": 0.0,
            "model": "slow",
            "endpoint": "/v1/chat/completions",
            "prompt_name": "short_short",
            "max_tokens": 8,
            "temperature": 0,
            "stream": True,
            "seed": 1,
        },
    ]
    async with httpx.AsyncClient(
        transport=ASGITransport(app), base_url="http://server", timeout=0.01
    ) as client:
        records = await run_trace(client, "http://server", rows)

    assert len(records) == 2
    assert all(record["error"] for record in records)


def test_failed_record_treats_incomplete_200_stream_as_failure():
    assert failed_record({"status": 200, "error": None, "stream_done": False})
    assert failed_record({"status": 200, "error": "broken", "stream_done": True})
    assert not failed_record({"status": 200, "error": None, "stream_done": True})