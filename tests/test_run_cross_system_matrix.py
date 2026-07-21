from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from run_cross_system_matrix import (  # noqa: E402
    parse_args,
    parse_system,
    run_one,
    validate_manifest_identity,
)


def manifest_rows() -> list[dict]:
    return [
        {
            "request_id": "r0",
            "scheduled_offset_s": 0.0,
            "model": "a",
            "endpoint": "/v1/chat/completions",
            "prompt_name": "short_short",
            "max_tokens": 2,
            "temperature": 0,
            "seed": 0,
            "stream": True,
        }
    ]


def test_parse_system_accepts_external_and_managed():
    external = parse_system("proposed=http://127.0.0.1:9000")
    assert external.name == "proposed"
    assert external.launch is None
    managed = parse_system(
        'cold=http://127.0.0.1:18100::["binary","--config","x"]'
    )
    assert managed.launch == ["binary", "--config", "x"]


def test_validate_manifest_identity_rejects_mismatch(tmp_path: Path):
    output = tmp_path / "rows.jsonl"
    row = {**manifest_rows()[0], "model": "b"}
    output.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_manifest_identity(manifest_rows(), output)


def test_run_one_writes_authenticated_output(monkeypatch, tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(manifest_rows()[0]) + "\n", encoding="utf-8")

    async def fake_run_trace(client, base_url, rows, timeout_s):
        return [
            {
                **rows[0],
                "status": 200,
                "error": None,
                "stream_done": True,
                "semantic_ttft_ms": 1.0,
                "completion_latency_ms": 2.0,
            }
        ]

    monkeypatch.setattr("run_cross_system_matrix.run_trace", fake_run_trace)
    output = tmp_path / "out.jsonl"
    result = asyncio.run(
        run_one(
            parse_system("test=http://127.0.0.1:1"),
            manifest,
            0,
            output,
            10,
            10,
            tmp_path / "run.log",
        )
    )
    assert result["return_code"] == 0
    assert result["failed"] == 0
    assert result["rows"] == 1
    assert result["output_sha256"]
    assert output.with_suffix(".run.json").exists()


def test_parse_args_rejects_non_positive_repeats():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--systems",
                "x=http://127.0.0.1:1",
                "--manifests",
                "trace.jsonl",
                "--repeats",
                "0",
                "--out-dir",
                "out",
            ]
        )
