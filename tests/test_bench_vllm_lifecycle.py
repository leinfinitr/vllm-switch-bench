#!/usr/bin/env python3
"""Tests for bench_vllm_lifecycle helpers.

These tests avoid launching vLLM. They guard the data-shaping pieces so later
changes to the benchmark harness do not silently corrupt result summaries.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
MODULE_PATH = SRC / "bench_vllm_lifecycle.py"
spec = importlib.util.spec_from_file_location("bench_vllm_lifecycle", MODULE_PATH)
assert spec is not None and spec.loader is not None
bench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bench
spec.loader.exec_module(bench)


def test_parse_args_rejects_unknown_prompt():
    try:
        bench.parse_args(["--model", "dummy", "--prompts", "missing_prompt"])
    except SystemExit as exc:
        assert "unknown prompts" in str(exc)
    else:
        raise AssertionError("parse_args should reject unknown prompt names")


def test_parse_args_rejects_removed_compat_sitecustomize_flag():
    try:
        bench.parse_args(["--model", "dummy", "--compat-sitecustomize", "legacy.py"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("legacy compat-sitecustomize flag should be removed")


def test_parse_args_default_out_dir_is_repo_local_results():
    args = bench.parse_args(["--model", "dummy"])
    assert args.out_dir == "results"


def test_write_summary_csv_flattens_nested_metrics(tmp_path):
    out = tmp_path / "summary.csv"
    bench.write_summary_csv(out, [
        {
            "run_id": "r1",
            "method": "sleep_l1",
            "model": "m",
            "prompt_name": "short_short",
            "repeat_index": 0,
            "ok": True,
            "startup_to_health_s": 1.5,
            "evict": {"latency_s": 0.25},
            "restore": {"latency_s": 0.75},
            "infer_before": {"ttft_s": 0.1, "client_latency_s": 0.8, "approx_tokens_per_s": 20.0},
            "infer_after": {"ttft_s": 0.2, "client_latency_s": 0.9, "approx_tokens_per_s": 18.0},
        }
    ])
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["method"] == "sleep_l1"
    assert rows[0]["evict_latency_s"] == "0.25"
    assert rows[0]["restore_latency_s"] == "0.75"
    assert rows[0]["ttft_after_s"] == "0.2"


def test_dry_run_creates_output_directory(tmp_path):
    rc = bench.main(["--model", "dummy", "--out-dir", str(tmp_path), "--dry-run"])
    assert rc == 0
    created = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(created) == 1
    assert (created[0] / "metadata.json").exists()
