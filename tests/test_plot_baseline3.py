from __future__ import annotations

import csv
import importlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def load_module():
    return importlib.import_module("plot_baseline3")


def test_load_summary_rows_and_aggregate_selected_methods(tmp_path: Path):
    csv_path = tmp_path / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "system",
                "method",
                "prompt_name",
                "ok",
                "evict_latency_s",
                "restore_latency_s",
                "latency_before_s",
                "latency_after_s",
                "memory_gpu_used_ready_mib",
                "memory_gpu_used_evict_mib",
                "tpot_before_s",
                "tpot_after_s",
            ]
        )
        writer.writerow(["vllm", "sleep_l1", "short_short", "True", 0.4, 0.1, 0.20, 0.08, 3000, 1500, 0.01, 0.02])
        writer.writerow(["vllm", "sleep_l1", "short_long", "True", 0.6, 0.3, 0.40, 0.12, 3200, 1600, 0.02, 0.04])
        writer.writerow(["serverless_llm", "delete_register", "short_short", "False", "", "", "", "", "", "", "", ""])
        writer.writerow(["swapserve_llm", "swapout_swapin", "short_short", "True", 0.45, 0.43, 0.09, 0.10, 3500, 900, 0.03, 0.04])

    mod = load_module()
    rows = mod.load_summary_rows(csv_path)
    agg = mod.aggregate_method_metrics(
        rows,
        methods=[("vllm", "sleep_l1"), ("swapserve_llm", "swapout_swapin")],
    )

    assert [row["label"] for row in agg] == ["vLLM Sleep L1", "SwapServeLLM"]
    assert agg[0]["count"] == 2
    assert agg[0]["evict_latency_s"] == pytest.approx(0.5)
    assert agg[0]["restore_latency_s"] == pytest.approx(0.2)
    assert agg[0]["latency_before_s"] == pytest.approx(0.3)
    assert agg[0]["latency_after_s"] == pytest.approx(0.1)
    assert agg[0]["memory_gpu_used_ready_mib"] == pytest.approx(3100.0)
    assert agg[0]["memory_gpu_used_evict_mib"] == pytest.approx(1550.0)
    assert agg[0]["tpot_before_s"] == pytest.approx(0.015)
    assert agg[0]["tpot_after_s"] == pytest.approx(0.03)
    assert agg[1]["count"] == 1
    assert agg[1]["restore_latency_s"] == pytest.approx(0.43)


def test_render_comparison_figure_writes_png(tmp_path: Path):
    mod = load_module()
    out_path = tmp_path / "comparison.png"
    mod.render_comparison_figure(
        [
            {
                "label": "vLLM Sleep L1",
                "count": 2,
                "evict_latency_s": 0.5,
                "restore_latency_s": 0.2,
                "latency_before_s": 0.3,
                "latency_after_s": 0.1,
                "memory_gpu_used_ready_mib": 3100.0,
                "memory_gpu_used_evict_mib": 1550.0,
                "tpot_before_s": 0.015,
                "tpot_after_s": 0.03,
            },
            {
                "label": "SwapServeLLM",
                "count": 1,
                "evict_latency_s": 0.45,
                "restore_latency_s": 0.43,
                "latency_before_s": 0.09,
                "latency_after_s": 0.10,
                "memory_gpu_used_ready_mib": 3500.0,
                "memory_gpu_used_evict_mib": 900.0,
                "tpot_before_s": 0.03,
                "tpot_after_s": 0.04,
            },
        ],
        out_path,
        title="Baseline3 comparison",
    )
    assert out_path.exists()
    assert out_path.stat().st_size > 0
