from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchlib.schema import JsonlLogger
from benchlib.sampling import Sampler, make_event, run_cmd


def test_run_cmd_captures_stdout():
    cp = run_cmd([sys.executable, "-c", "print('ok')"], timeout=5)
    assert cp.stdout.strip() == "ok"


def test_make_event_uses_context_and_metrics(monkeypatch):
    monkeypatch.setattr("benchlib.sampling.query_gpu", lambda: {"gpu_used_mib": 10, "gpu_free_mib": 20, "gpu_util_pct": 30})
    monkeypatch.setattr(
        "benchlib.sampling.query_cpu",
        lambda pid: {"cpu_used_mib": 40, "cpu_available_mib": 50, "proc_rss_mib": 60, "proc_uss_mib": 70},
    )
    event = make_event(
        {"system": "vllm", "run_id": "r1", "method": "sleep_l1", "model": "m", "prompt_name": "short_short", "repeat_index": 0},
        "sample",
        start_ts=time.time() - 1,
        pid=123,
    )
    assert event.system == "vllm"
    assert event.proc_pid == 123
    assert event.gpu_used_mib == 10
    assert event.proc_uss_mib == 70


def test_sampler_writes_sample_events(tmp_path):
    logger = JsonlLogger(tmp_path / "events.jsonl")
    ctx = {"system": "vllm", "run_id": "r1", "method": "sleep_l1", "model": "m", "prompt_name": "short_short", "repeat_index": 0}
    with Sampler(logger, ctx, time.time(), lambda: None, interval_s=0.01):
        time.sleep(0.03)
    logger.close()
    lines = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len(lines) >= 1
    assert all(line["event"] == "sample" for line in lines)
