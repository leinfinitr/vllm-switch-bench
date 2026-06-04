from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench_baseline3 import (
    DEFAULT_VLLM_RESULT,
    build_serverless_cmd,
    make_blocker_row,
    normalize_rows,
    read_summary_rows,
)


def test_build_serverless_command_uses_registered_model_name_when_present():
    cmd = build_serverless_cmd(
        repo="/repo/ServerlessLLM",
        model="/models/hf/Qwen2.5-0.5B-Instruct",
        registered_model_name="qwen2p5-0p5b",
        base_url="http://127.0.0.1:8343",
        prompts=["short_short"],
        repeats=1,
        out_dir="results/tmp/serverless",
    )
    joined = " ".join(cmd)
    assert "src/bench_serverless_llm.py" in joined
    assert "--registered-model-name" in cmd
    assert "qwen2p5-0p5b" in cmd

def test_build_serverless_command_can_run_multiple_methods():
    cmd = build_serverless_cmd(
        repo="/repo/ServerlessLLM",
        model="/models/hf/Qwen2.5-0.5B-Instruct",
        registered_model_name="qwen2p5-0p5b",
        base_url="http://127.0.0.1:8343",
        prompts=["short_short"],
        repeats=1,
        out_dir="results/tmp/serverless",
        methods=["delete_register", "scale_to_zero_restore"],
    )

    idx = cmd.index("--methods")
    assert cmd[idx + 1 : idx + 3] == ["delete_register", "scale_to_zero_restore"]


def test_default_vllm_result_points_to_latest_simplified_schema_run():
    assert str(DEFAULT_VLLM_RESULT) == "results/baselines/vllm/qwen2p5_0p5b/20260603_150331"


def test_normalize_rows_injects_default_system_for_legacy_rows():
    rows = [{"method": "cold_reload", "model": "m", "prompt_name": "short_short", "repeat_index": 0, "ok": True}]
    normalized = normalize_rows(rows, default_system="vllm")
    assert normalized[0]["system"] == "vllm"



def test_make_blocker_row_marks_unsupported():
    row = make_blocker_row(
        system="swapserve_llm",
        method="swapout_swapin",
        model="qwen",
        prompt_name="short_short",
        repeat_index=0,
        error="podman missing",
    )
    assert row["system"] == "swapserve_llm"
    assert row["unsupported"] is True
    assert row["ok"] is False
    assert "podman" in row["error"]



def test_read_summary_rows_loads_json(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps([{"system": "vllm", "ok": True}]), encoding="utf-8")
    rows = read_summary_rows(tmp_path)
    assert rows == [{"system": "vllm", "ok": True}]
