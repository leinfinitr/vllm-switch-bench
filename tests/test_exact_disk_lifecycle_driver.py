from __future__ import annotations

import json
import sys
from pathlib import Path

from llm_switch_bench.experiments.exact_disk import lifecycle_driver


def test_lifecycle_driver_records_repeated_cycle_latencies(monkeypatch, tmp_path: Path):
    observation = tmp_path / "observation.json"
    monkeypatch.setenv("LLM_SWITCH_BENCH_OUTPUT_OBSERVATION", str(observation))
    monkeypatch.setattr(
        sys,
        "argv",
        ["lifecycle_driver", "--cycles", "3", "--served-model-name", "model-a"],
    )
    monkeypatch.setattr(lifecycle_driver, "wait_ready", lambda *_args: None)

    infer_results = iter(
        [
            {"text": "ok", "finish_reason": "stop"},
            {"text": "ok", "finish_reason": "stop"},
            {"text": "ok", "finish_reason": "stop"},
            {"text": "ok", "finish_reason": "stop"},
        ]
    )
    monkeypatch.setattr(lifecycle_driver, "infer", lambda *_args: next(infer_results))

    calls: list[str] = []

    def fake_post(url: str, payload, _timeout_s: float):
        calls.append(url)
        if url.endswith("/collective_rpc"):
            return {
                "results": [
                    {
                        "released_bytes_total": 4,
                        "remaining_cpu_backup_bytes": 0,
                        "pending_release_bytes": 0,
                    }
                ]
            }
        return None

    monkeypatch.setattr(lifecycle_driver, "post_json", fake_post)
    ticks = iter(float(index) for index in range(20))
    monkeypatch.setattr(lifecycle_driver.time, "perf_counter", lambda: next(ticks))

    assert lifecycle_driver.main() == 0
    result = json.loads(observation.read_text(encoding="utf-8"))
    assert result["demotion_latency_s"] == 1.0
    assert len(result["cycles"]) == 3
    assert all(cycle["output_equal"] for cycle in result["cycles"])
    assert [cycle["sleep_latency_s"] for cycle in result["cycles"]] == [1.0, 1.0, 1.0]
    assert [cycle["wake_latency_s"] for cycle in result["cycles"]] == [1.0, 1.0, 1.0]
    assert sum(url.endswith("/sleep?level=1") for url in calls) == 3
    assert sum(url.endswith("/wake_up") for url in calls) == 3


def test_lifecycle_driver_rejects_non_positive_cycles(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("LLM_SWITCH_BENCH_OUTPUT_OBSERVATION", str(tmp_path / "observation.json"))
    monkeypatch.setattr(sys, "argv", ["lifecycle_driver", "--cycles", "0"])

    try:
        lifecycle_driver.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected argparse failure")
