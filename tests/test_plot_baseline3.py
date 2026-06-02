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
                "tokens_per_s_before",
                "tokens_per_s_after",
            ]
        )
        writer.writerow(["vllm", "sleep_l1", "short_short", "True", 0.4, 0.1, 0.20, 0.08, 100, 200])
        writer.writerow(["vllm", "sleep_l1", "short_long", "True", 0.6, 0.3, 0.40, 0.12, 120, 220])
        writer.writerow(["serverless_llm", "delete_register", "short_short", "False", "", "", "", "", "", ""])
        writer.writerow(["swapserve_llm", "swapout_swapin", "short_short", "True", 0.45, 0.43, 0.09, 0.10, 340, 330])

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
    assert agg[0]["tokens_per_s_before"] == pytest.approx(110.0)
    assert agg[0]["tokens_per_s_after"] == pytest.approx(210.0)
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
                "tokens_per_s_before": 110.0,
                "tokens_per_s_after": 210.0,
            },
            {
                "label": "SwapServeLLM",
                "count": 1,
                "evict_latency_s": 0.45,
                "restore_latency_s": 0.43,
                "latency_before_s": 0.09,
                "latency_after_s": 0.10,
                "tokens_per_s_before": 340.0,
                "tokens_per_s_after": 330.0,
            },
        ],
        out_path,
        title="Baseline3 comparison",
    )
    assert out_path.exists()
    assert out_path.stat().st_size > 0
