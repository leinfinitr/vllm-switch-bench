import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tool.analyze_request_switch import summarize


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
