import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src/tool/analyze_cross_system.py"
spec = importlib.util.spec_from_file_location("analyze_cross_system", MODULE)
assert spec and spec.loader
analyze = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = analyze
spec.loader.exec_module(analyze)


def test_percentile_interpolates():
    assert analyze.percentile([1.0, 3.0], 0.5) == pytest.approx(2.0)


def test_summarize_lifecycle_uses_only_ok_rows(tmp_path: Path):
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            [
                {
                    "method": "sleep_l1",
                    "ok": True,
                    "evict": {"latency_s": 0.1},
                    "restore": {"latency_s": 0.2},
                    "memory_gpu_used_ready_mib": 1000,
                    "memory_gpu_used_evict_mib": 100,
                },
                {
                    "method": "sleep_l1",
                    "ok": False,
                    "evict": {"latency_s": 99},
                    "restore": {"latency_s": 99},
                },
            ]
        ),
        encoding="utf-8",
    )
    summary = analyze.summarize_lifecycle(path, 1, 100)
    assert summary["sleep_l1"]["runs"] == 2
    assert summary["sleep_l1"]["success"] == 1
    assert summary["sleep_l1"]["activation_ms"]["median"] == pytest.approx(300)


def test_trace_analyzer_keeps_failure_in_denominator(tmp_path: Path):
    output = tmp_path / "sys-trace-r0.jsonl"
    rows = [
        {
            "request_id": "a",
            "status": 200,
            "error": None,
            "stream_done": True,
            "semantic_ttft_ms": 10,
            "completion_latency_ms": 20,
        },
        {
            "request_id": "b",
            "status": 200,
            "error": "broken",
            "stream_done": False,
            "semantic_ttft_ms": 999,
            "completion_latency_ms": 999,
        },
    ]
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    import hashlib

    matrix = [
        {
            "system": "sys",
            "manifest": "trace.jsonl",
            "output": output.name,
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
    ]
    (tmp_path / "matrix.json").write_text(json.dumps(matrix))
    summary = analyze.summarize_trace_dir(tmp_path, 1, 100)
    cell = summary["sys:trace.jsonl"]
    assert cell["requests"] == 2
    assert cell["success"] == 1
    assert cell["failure_rate"] == pytest.approx(0.5)
    assert cell["pooled_ttft_ms"]["median"] == pytest.approx(10)


def test_trace_group_name_includes_parent_to_avoid_collisions():
    path = Path("results/cross_system/raw/proposed/request-traces-final")
    assert analyze.trace_group_name(path) == "proposed/request-traces-final"


def test_metric_summary_bootstrap_is_deterministic():
    first = analyze.metric_summary([1.0, 2.0, 3.0], 7, 100)
    second = analyze.metric_summary([1.0, 2.0, 3.0], 7, 100)
    assert first == second
    assert first["median"] == 2.0
