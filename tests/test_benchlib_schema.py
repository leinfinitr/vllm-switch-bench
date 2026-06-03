from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchlib.http import parse_openai_stream_response
from benchlib.schema import Event, JsonlLogger, PROMPTS, write_summary_csv


class FakeResponse:
    def __init__(self, lines: list[str]):
        self._lines = lines

    def iter_lines(self, decode_unicode: bool = True):
        assert decode_unicode is True
        yield from self._lines


def test_summary_csv_includes_system_and_method(tmp_path):
    out = tmp_path / "summary.csv"
    write_summary_csv(
        out,
        [
            {
                "system": "vllm",
                "run_id": "r1",
                "method": "sleep_l1",
                "model": "m",
                "prompt_name": "short_short",
                "repeat_index": 0,
                "ok": True,
                "startup_to_health_s": 1.25,
                "evict": {"latency_s": 0.2},
                "restore": {"latency_s": 0.4},
                "infer_before": {"ttft_s": 0.1, "client_latency_s": 0.8, "approx_tokens_per_s": 20.0},
                "infer_after": {"ttft_s": 0.2, "client_latency_s": 0.9, "approx_tokens_per_s": 18.0},
            }
        ],
    )
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["system"] == "vllm"
    assert rows[0]["method"] == "sleep_l1"


def test_prompt_catalog_has_three_shapes():
    assert set(PROMPTS) == {"short_short", "long_short", "short_long"}
    assert PROMPTS["long_short"]["max_tokens"] == 24
    assert PROMPTS["short_long"]["max_tokens"] == 160


def test_openai_stream_parser_extracts_ttft_and_text():
    response = FakeResponse(
        [
            'data: {"choices": [{"text": "Hello"}]}',
            'data: {"choices": [{"delta": {"content": " world"}}], "usage": {"completion_tokens": 2}}',
            'data: [DONE]',
        ]
    )
    now_values = iter([1.125])
    parsed = parse_openai_stream_response(response, started_at=1.0, now_fn=lambda: next(now_values))
    assert parsed["ttft_s"] == 0.125
    assert parsed["output_text"] == "Hello world"
    assert parsed["completion_tokens"] == 2


def test_event_logger_writes_jsonl(tmp_path):
    path = tmp_path / "events.jsonl"
    logger = JsonlLogger(path)
    logger.write(
        Event(
            system="vllm",
            run_id="r1",
            method="sleep_l1",
            model="m",
            prompt_name="short_short",
            repeat_index=0,
            event="run_start",
            ts=1.0,
            elapsed_s=0.0,
        )
    )
    logger.close()
    data = json.loads(path.read_text().strip())
    assert data["system"] == "vllm"
    assert data["event"] == "run_start"


def test_summary_csv_uses_baseline_metric_schema(tmp_path):
    out = tmp_path / "summary.csv"
    write_summary_csv(
        out,
        [
            {
                "system": "vllm",
                "method": "sleep_l1",
                "model": "qwen",
                "prompt_name": "short_short",
                "repeat_index": 0,
                "ok": True,
                "startup_latency_s": 1.5,
                "memory_gpu_used_ready_mib": 4000,
                "memory_cpu_used_ready_mib": 12000,
                "evict": {"latency_s": 0.2},
                "restore": {"latency_s": 0.1},
                "memory_gpu_used_evict_mib": 900,
                "memory_cpu_used_evict_mib": 13000,
                "infer_before": {
                    "ttft_s": 0.05,
                    "client_latency_s": 0.25,
                    "completion_tokens": 11,
                },
                "infer_after": {
                    "ttft_s": 0.04,
                    "client_latency_s": 0.24,
                    "completion_tokens": 11,
                },
            }
        ],
    )
    rows = list(csv.DictReader(out.open()))
    row = rows[0]
    header = out.read_text().splitlines()[0]
    assert "startup_latency_s" in header
    assert "startup_to_health_s" not in header
    assert "memory_gpu_used_ready_mib" in header
    assert "memory_gpu_used_evict_mib" in header
    assert "tpot_before_s" in header
    assert "tokens_per_s_before" not in header
    assert row["output_tokens_before"] == "11"
    assert row["tpot_before_s"] == "0.02"
    assert row["ttft_available"] == "True"
    assert row["tpot_available"] == "True"
