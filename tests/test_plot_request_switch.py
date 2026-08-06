from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from llm_switch_bench.experiments.request_driven_switch.plot import plot_summary  # noqa: E402


def test_plot_summary_creates_expected_figures(tmp_path):
    summary = {
        "w0": {
            "semantic_ttft_ms": {"median": 10, "p95": 15},
            "completion_latency_ms": {"median": 20, "p95": 25},
        },
        "w1": {
            "semantic_ttft_ms": {"median": 100, "p95": 130},
            "completion_latency_ms": {"median": 200, "p95": 240},
        },
        "w2": {
            "semantic_ttft_ms": {"median": 30, "p95": 110},
            "completion_latency_ms": {"median": 80, "p95": 200},
        },
        "profile_ablation": {
            "1.5b": {
                "first_miss": {"latency_s": 1.0},
                "clean_reuse_latency_s_median": 0.1,
                "wake_latency_s_median": 0.3,
            }
        },
        "pressure": {
            "p1": {
                "client_rss_delta_bytes": {"a": -4_000_000_000},
                "memavailable_delta_bytes": 4_100_000_000,
            }
        },
        "controller": {
            "sleep_latency_ms": {"median": 100},
            "wake_latency_ms": {"median": 400},
            "request_drain_ms": {"median": 1},
        },
    }
    outputs = plot_summary(summary, tmp_path)
    assert {path.name for path in outputs} == {
        "request-workloads.png",
        "switch-breakdown.png",
        "backup-ablation.png",
        "physical-reclaim.png",
    }
    assert all(path.stat().st_size > 0 for path in outputs)
