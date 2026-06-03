from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze_baseline3 import build_report, group_rows


def test_baseline3_report_groups_by_system_method_prompt():
    rows = [
        {"system": "vllm", "method": "sleep_l1", "prompt_name": "short_short", "ok": True, "restore": {"latency_s": 0.1}},
        {"system": "vllm", "method": "sleep_l1", "prompt_name": "short_short", "ok": True, "restore": {"latency_s": 0.2}},
        {"system": "serverless_llm", "method": "delete_register", "prompt_name": "short_short", "ok": False, "unsupported": True, "error": "blocker"},
    ]
    grouped = group_rows(rows)
    assert ("vllm", "sleep_l1", "short_short") in grouped
    assert len(grouped[("vllm", "sleep_l1", "short_short")]) == 2



def test_report_marks_unsupported_rows_separately(tmp_path: Path):
    rows = [
        {"system": "swapserve_llm", "method": "swapout_swapin", "prompt_name": "short_short", "ok": False, "unsupported": True, "error": "podman missing"}
    ]
    report = build_report(rows, metadata={"model": "qwen"})
    assert "Unsupported / blocked rows" in report
    assert "podman missing" in report



def test_stage_breakdown_parser_handles_missing_logs():
    rows = [
        {"system": "swapserve_llm", "method": "swapout_swapin", "prompt_name": "short_short", "ok": True, "stage_breakdown": {"swapout.pause_container_s": 0.01}}
    ]
    report = build_report(rows, metadata={"model": "qwen"})
    assert "swapout.pause_container_s" in report


def test_report_includes_startup_latency_and_estimated_restore_note():
    rows = [
        {
            "system": "serverless_llm",
            "method": "scale_to_zero_restore",
            "prompt_name": "short_short",
            "ok": True,
            "startup_latency_s": 0.6,
            "restore_latency_estimated": True,
            "restore": {"latency_s": 10.0},
            "evict": {"latency_s": 2.5},
        }
    ]
    report = build_report(rows, metadata={"model": "qwen"})
    assert "mean_startup_s" in report
    assert "restore_latency_estimated" in report
    assert "startup_to_health_s" not in report
