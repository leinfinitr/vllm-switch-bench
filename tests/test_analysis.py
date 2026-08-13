import hashlib
import json
from pathlib import Path

import pytest

from llm_switch_bench.common.traces import write_manifest
from llm_switch_bench import analysis as analyze


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
    manifest = tmp_path / "trace.jsonl"
    manifest_rows = [
        {
            "request_id": request_id,
            "scheduled_offset_s": index,
            "model": "model-a",
            "endpoint": "/v1/chat/completions",
            "prompt_name": "short_short",
            "max_tokens": 2,
            "temperature": 0,
            "stream": True,
            "seed": 1,
        }
        for index, request_id in enumerate(["a", "b"])
    ]
    write_manifest(manifest, manifest_rows)
    output = tmp_path / "sys-trace-r0.jsonl"
    rows = [
        {
            **manifest_rows[0],
            "status": 200,
            "error": None,
            "stream_done": True,
            "semantic_ttft_ms": 10,
            "completion_latency_ms": 20,
            "output_text": "ok",
        },
        {
            **manifest_rows[1],
            "status": 200,
            "error": "broken",
            "stream_done": False,
            "semantic_ttft_ms": 999,
            "completion_latency_ms": 999,
            "output_text": "partial",
        },
    ]
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))

    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    matrix = [
        {
            "system": "sys",
            "manifest": manifest.name,
            "manifest_sha256": manifest_sha,
            "repeat": 0,
            "rows": 2,
            "return_code": 0,
            "output": output.name,
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
    ]
    (tmp_path / "matrix.json").write_text(json.dumps(matrix))
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "repeats": 1,
                "systems": [{"name": "sys"}],
                "manifests": [{"path": str(manifest), "sha256": manifest_sha}],
            }
        )
    )
    summary = analyze.summarize_trace_dir(tmp_path, 1, 100)
    cell = summary['["sys","trace.jsonl"]']
    assert cell["requests"] == 2
    assert cell["success"] == 1
    assert cell["failure_rate"] == pytest.approx(0.5)
    assert cell["pooled_ttft_ms"]["median"] == pytest.approx(10)


def test_trace_analyzer_rejects_incomplete_matrix(tmp_path: Path):
    manifest = tmp_path / "trace.jsonl"
    write_manifest(
        manifest,
        [
            {
                "request_id": "a",
                "scheduled_offset_s": 0,
                "model": "model-a",
                "endpoint": "/v1/chat/completions",
                "prompt_name": "short_short",
                "max_tokens": 2,
                "temperature": 0,
                "stream": True,
                "seed": 1,
            }
        ],
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "repeats": 1,
                "systems": [{"name": "sys"}],
                "manifests": [{"path": str(manifest), "sha256": manifest_sha}],
            }
        )
    )
    (tmp_path / "matrix.json").write_text("[]")
    with pytest.raises(ValueError, match="matrix incomplete"):
        analyze.summarize_trace_dir(tmp_path, 1, 100)


def test_trace_analyzer_rejects_full_dispatch_semantics_mismatch(tmp_path: Path):
    manifest = tmp_path / "trace.jsonl"
    expected = {
        "request_id": "a",
        "scheduled_offset_s": 0,
        "model": "model-a",
        "endpoint": "/v1/chat/completions",
        "prompt_name": "short_short",
        "max_tokens": 2,
        "temperature": 0,
        "stream": True,
        "seed": 1,
    }
    write_manifest(manifest, [expected])
    observed = {
        **expected,
        "max_tokens": 99,
        "status": 200,
        "error": None,
        "stream_done": True,
        "semantic_ttft_ms": 1,
        "completion_latency_ms": 2,
        "output_text": "ok",
    }
    output = tmp_path / "sys-trace-r0.jsonl"
    output.write_text(json.dumps(observed) + "\n")
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "repeats": 1,
                "systems": [{"name": "sys"}],
                "manifests": [{"path": str(manifest), "sha256": manifest_sha}],
            }
        )
    )
    (tmp_path / "matrix.json").write_text(
        json.dumps(
            [
                {
                    "system": "sys",
                    "manifest": manifest.name,
                    "manifest_sha256": manifest_sha,
                    "repeat": 0,
                    "rows": 1,
                    "return_code": 0,
                    "output": output.name,
                    "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="request identity mismatch"):
        analyze.summarize_trace_dir(tmp_path, 1, 100)


def test_trace_analyzer_rejects_nonzero_run(tmp_path: Path):
    manifest = tmp_path / "trace.jsonl"
    row = {
        "request_id": "a",
        "scheduled_offset_s": 0,
        "model": "model-a",
        "endpoint": "/v1/chat/completions",
        "prompt_name": "short_short",
        "max_tokens": 2,
        "temperature": 0,
        "stream": True,
        "seed": 1,
    }
    write_manifest(manifest, [row])
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "repeats": 1,
                "systems": [{"name": "sys"}],
                "manifests": [{"path": str(manifest), "sha256": manifest_sha}],
            }
        )
    )
    (tmp_path / "matrix.json").write_text(
        json.dumps(
            [
                {
                    "system": "sys",
                    "manifest": manifest.name,
                    "repeat": 0,
                    "return_code": 1,
                    "output": "missing.jsonl",
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="nonzero harness return code"):
        analyze.summarize_trace_dir(tmp_path, 1, 100)


def test_trace_analyzer_rejects_empty_system_or_manifest_declarations(tmp_path: Path):
    (tmp_path / "metadata.json").write_text(
        json.dumps({"systems": [], "manifests": [], "repeats": 1}),
        encoding="utf-8",
    )
    (tmp_path / "matrix.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        analyze.summarize_trace_dir(tmp_path, 1, 100)


def test_request_identity_rejects_partial_full_identity():
    expected = {
        "request_id": "a",
        "scheduled_offset_s": 0,
        "model": "model-a",
        "endpoint": "/v1/chat/completions",
        "prompt_name": "short_short",
        "max_tokens": 2,
        "temperature": 0,
        "stream": True,
        "seed": 1,
    }
    observed = {**expected, "max_tokens": 999}
    observed.pop("seed")
    with pytest.raises(ValueError, match="partial dispatch identity"):
        analyze.request_identity_matches(expected, observed)


def test_trace_group_name_includes_parent_to_avoid_collisions():
    path = Path("results/cross_system/raw/proposed/request-traces-final")
    assert analyze.trace_group_name(path) == "proposed/request-traces-final"


def test_metric_summary_bootstrap_is_deterministic():
    first = analyze.metric_summary([1.0, 2.0, 3.0], 7, 100)
    second = analyze.metric_summary([1.0, 2.0, 3.0], 7, 100)
    assert first == second
    assert first["median"] == 2.0
