#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "bench_vllm_offline.py"
spec = importlib.util.spec_from_file_location("bench_vllm_offline", MODULE_PATH)
assert spec is not None and spec.loader is not None
bench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bench
spec.loader.exec_module(bench)


def test_offline_prompt_catalog_has_expected_prompts():
    assert {"short_short", "long_short", "short_long"}.issubset(bench.PROMPTS)
    assert bench.PROMPTS["short_short"]["max_tokens"] == 32


def test_offline_summary_csv_flattens_rows(tmp_path):
    out = tmp_path / "summary.csv"
    bench.write_summary_csv(out, [
        {
            "run_id": "r1",
            "method": "sleep_l1",
            "model": "m",
            "prompt_name": "short_short",
            "repeat_index": 0,
            "ok": True,
            "startup_to_ready_s": 1.0,
            "evict": {"latency_s": 0.2},
            "restore": {"latency_s": 0.3},
            "infer_before": {"client_latency_s": 0.4, "approx_tokens_per_s": 10.0},
            "infer_after": {"client_latency_s": 0.5, "approx_tokens_per_s": 9.0},
        }
    ])
    text = out.read_text()
    assert "sleep_l1" in text
    assert "0.2" in text
    assert "0.3" in text
