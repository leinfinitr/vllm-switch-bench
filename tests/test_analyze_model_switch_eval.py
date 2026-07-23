from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/analyze_model_switch_eval.py"
SPEC = importlib.util.spec_from_file_location("analyze_model_switch_eval", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def valid_record(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "status": 200,
        "error": None,
        "stream_done": True,
        "semantic_ttft_ms": 10.0,
        "output_text": "ok",
        "client_dispatch_offset_s": 0.0,
        "completion_latency_ms": 100.0,
    }
    row.update(overrides)
    return row


def write_trace(tmp_path: Path, row: dict[str, object]) -> Path:
    path = tmp_path / "test-trace.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": 302},
        {"status": None},
        {"error": "SSE error"},
        {"stream_done": False},
        {"semantic_ttft_ms": None},
        {"output_text": ""},
    ],
)
def test_trace_rejects_non_strict_rows(tmp_path: Path, overrides: dict[str, object]) -> None:
    path = write_trace(tmp_path, valid_record(**overrides))
    with pytest.raises(ValueError, match="strict failures"):
        MODULE.trace("test", [path])


def test_trace_accepts_strict_2xx_row(tmp_path: Path) -> None:
    path = write_trace(tmp_path, valid_record(status=201))
    result = MODULE.trace("test", [path])
    assert result["success_requests"] == 1
    assert result["attempted_requests"] == 1
