import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tool.analyze_request_switch import (  # noqa: E402
    summarize,
    summarize_controller_switches,
)


def test_summarize_request_switch_keeps_failures(tmp_path):
    (tmp_path / "w1-r0.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"status": 200, "semantic_ttft_ms": 10, "completion_latency_ms": 20}),
                json.dumps({"status": 500, "error": "x", "completion_latency_ms": 30}),
            ]
        )
    )
    summary = summarize(tmp_path)
    assert summary["w1"]["requests"] == 2
    assert summary["w1"]["success"] == 1
    assert summary["w1"]["failed"] == 1
    assert summary["w1"]["semantic_ttft_ms"]["median"] == 10


def test_summarize_controller_switches_separates_hits_and_switches(tmp_path):
    path = tmp_path / "events.jsonl"
    rows = [
        {
            "path": "/v1/chat/completions",
            "switch_needed": True,
            "switch_latency_ms": 600,
            "sleep_latency_ms": 100,
            "wake_latency_ms": 400,
            "request_drain_ms": 0,
        },
        {"path": "/v1/chat/completions", "switch_needed": False},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = summarize_controller_switches(path)
    assert summary["requests"] == 2
    assert summary["switches"] == 1
    assert summary["steady_hits"] == 1
    assert summary["switch_latency_ms"]["median"] == 600
    assert summary["sleep_latency_ms"]["median"] == 100
