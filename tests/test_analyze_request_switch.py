import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tool.analyze_request_switch import (  # noqa: E402
    summarize,
    summarize_controller_switches,
    summarize_manifest,
)


def test_summarize_request_switch_keeps_failures(tmp_path):
    (tmp_path / "w1-r0.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "status": 200,
                        "error": None,
                        "stream_done": True,
                        "semantic_ttft_ms": 10,
                        "completion_latency_ms": 20,
                    }
                ),
                json.dumps({"status": 500, "error": "x", "completion_latency_ms": 30}),
            ]
        )
    )
    manifest = tmp_path / "request-switch-alternating.jsonl"
    manifest.write_text("{}\n{}\n")
    digest = __import__("hashlib").sha256(manifest.read_bytes()).hexdigest()
    (tmp_path / "metadata.json").write_text(json.dumps({
        "repeats": 1, "workloads": ["w1"], "runs": [{
            "workload": "w1", "repeat": 0,
            "manifest": manifest.name, "manifest_sha256": digest,
            "output": "w1-r0.jsonl", "returncode": 0,
        }]
    }))
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


def test_summarize_manifest_reports_scheduled_rate(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"request_id": "r1", "scheduled_offset_s": 0}),
                json.dumps({"request_id": "r2", "scheduled_offset_s": 2}),
                json.dumps({"request_id": "r3", "scheduled_offset_s": 4}),
            ]
        )
    )
    assert summarize_manifest(path) == {
        "requests": 3,
        "scheduled_duration_s": 4.0,
        "offered_rate_rps": 0.5,
    }


def test_summarize_rejects_missing_or_failed_matrix_runs(tmp_path):
    manifest = tmp_path / "request-switch-steady.jsonl"
    manifest.write_text("{}\n{}\n")
    (tmp_path / "w0-r0.jsonl").write_text(
        json.dumps({"status": 200, "stream_done": True}) + "\n"
    )
    digest = __import__("hashlib").sha256(manifest.read_bytes()).hexdigest()
    metadata = {
        "repeats": 2,
        "runs": [
            {
                "workload": "w0",
                "repeat": 0,
                "manifest": "request-switch-steady.jsonl",
                "manifest_sha256": digest,
                "output": "w0-r0.jsonl",
                "returncode": 0,
            },
            {
                "workload": "w0",
                "repeat": 1,
                "manifest": "request-switch-steady.jsonl",
                "manifest_sha256": digest,
                "output": "w0-r1.jsonl",
                "returncode": 1,
            },
        ],
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="incomplete benchmark matrix"):
        summarize(tmp_path)


def test_summarize_rejects_metadata_missing_an_entire_workload(tmp_path):
    manifest = tmp_path / "request-switch-steady.jsonl"
    manifest.write_text("{}\n")
    digest = __import__("hashlib").sha256(manifest.read_bytes()).hexdigest()
    runs = []
    for repeat in range(2):
        output = f"w0-r{repeat}.jsonl"
        (tmp_path / output).write_text(
            json.dumps({"status": 200, "stream_done": True}) + "\n"
        )
        runs.append(
            {
                "workload": "w0",
                "repeat": repeat,
                "manifest": "request-switch-steady.jsonl",
                "manifest_sha256": digest,
                "output": output,
                "returncode": 0,
            }
        )
    (tmp_path / "metadata.json").write_text(
        json.dumps({"repeats": 2, "workloads": ["w0", "w1"], "runs": runs})
    )
    with pytest.raises(ValueError, match="missing run w1-r0"):
        summarize(tmp_path)


def test_summarize_requires_metadata_and_rejects_undeclared_output(tmp_path):
    (tmp_path / "w1-r0.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="metadata"):
        summarize(tmp_path)


def test_summarize_rejects_undeclared_output_checksum_and_redirect(tmp_path):
    manifest = tmp_path / "trace.jsonl"
    manifest.write_text("{}\n")
    digest = __import__("hashlib").sha256(manifest.read_bytes()).hexdigest()
    row = {"status": 302, "error": None, "stream_done": True}
    (tmp_path / "w0-r0.jsonl").write_text(json.dumps(row) + "\n")
    metadata = {"repeats": 1, "workloads": ["w0"], "runs": [{
        "workload": "w0", "repeat": 0, "manifest": manifest.name,
        "manifest_sha256": digest, "output": "w0-r0.jsonl", "returncode": 0,
    }]}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    summary = summarize(tmp_path)
    assert summary["w0"]["success"] == 0
    (tmp_path / "w0-r99.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="undeclared output"):
        summarize(tmp_path)
    (tmp_path / "w0-r99.jsonl").unlink()
    manifest.write_text("{\"changed\": true}\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        summarize(tmp_path)


def test_summarize_binds_each_output_row_to_manifest_identity(tmp_path):
    manifest = tmp_path / "trace.jsonl"
    manifest.write_text(
        json.dumps(
            {"request_id": "expected", "model": "a", "scheduled_offset_s": 1.0}
        )
        + "\n"
    )
    digest = __import__("hashlib").sha256(manifest.read_bytes()).hexdigest()
    output = tmp_path / "w0-r0.jsonl"
    output.write_text(
        json.dumps(
            {
                "request_id": "swapped",
                "model": "b",
                "scheduled_offset_s": 2.0,
                "status": 200,
                "error": None,
                "stream_done": True,
            }
        )
        + "\n"
    )
    run = {
        "workload": "w0",
        "repeat": 0,
        "manifest": manifest.name,
        "manifest_sha256": digest,
        "output": output.name,
        "returncode": 0,
    }
    (tmp_path / "metadata.json").write_text(
        json.dumps({"repeats": 1, "workloads": ["w0"], "runs": [run]})
    )
    with pytest.raises(ValueError, match="mismatches manifest field request_id"):
        summarize(tmp_path)

    run2 = {**run, "workload": "w1", "output": f"./{output.name}"}
    (tmp_path / "metadata.json").write_text(
        json.dumps({"repeats": 1, "workloads": ["w0", "w1"], "runs": [run, run2]})
    )
    with pytest.raises(ValueError, match="duplicate output path"):
        summarize(tmp_path)
