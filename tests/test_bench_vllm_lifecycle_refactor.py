from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
MODULE_PATH = SRC / "bench_vllm_lifecycle.py"
spec = importlib.util.spec_from_file_location("bench_vllm_lifecycle", MODULE_PATH)
assert spec is not None and spec.loader is not None
bench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bench
spec.loader.exec_module(bench)


def test_vllm_harness_uses_benchlib_prompt_catalog():
    from benchlib.schema import PROMPTS

    assert bench.PROMPTS is PROMPTS


def test_vllm_harness_uses_benchlib_summary_writer(tmp_path):
    out = tmp_path / "summary.csv"
    bench.write_summary_csv(
        out,
        [
            {
                "system": "vllm",
                "run_id": "r1",
                "method": "sleep_l1",
                "model": "m",
                "prompt_name": "short_short",
                "repeat_index": 0,
                "ok": True,
                "infer_before": {"ttft_s": 0.1, "client_latency_s": 0.2, "approx_tokens_per_s": 3.0},
                "infer_after": {"ttft_s": 0.2, "client_latency_s": 0.3, "approx_tokens_per_s": 4.0},
            }
        ],
    )
    text = out.read_text()
    assert text.splitlines()[0].startswith("system,")
    assert ",vllm," in text or text.endswith(",vllm") or "vllm" in text


def test_vllm_main_dry_run_metadata_records_system(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "run_cmd", lambda *args, **kwargs: type("CP", (), {"stdout": "0,FakeGPU,10240,999.1\n"})())
    rc = bench.main(["--model", "dummy", "--out-dir", str(tmp_path), "--dry-run"])
    assert rc == 0
    created = [p for p in tmp_path.iterdir() if p.is_dir()]
    metadata = (created[0] / "metadata.json").read_text()
    assert '"system": "vllm"' in metadata


def test_parse_args_rejects_unknown_prompt():
    with pytest.raises(SystemExit) as exc:
        bench.parse_args(["--model", "dummy", "--prompts", "missing_prompt"])
    assert "unknown prompts" in str(exc.value)


def test_collect_sleep_profile_window_filters_by_monotonic_time(tmp_path):
    profile = tmp_path / "sleep_profile.jsonl"
    profile.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"monotonic_s": 0.5, "phase": "too_early", "latency_s": 1.0},
                {"monotonic_s": 1.5, "phase": "allocator_sleep", "latency_s": 0.2},
                {"monotonic_s": 2.5, "phase": "too_late", "latency_s": 3.0},
            ]
        ),
        encoding="utf-8",
    )

    window = bench.collect_sleep_profile_window(profile, "sleep", 1.0, 2.0)

    assert window["operation"] == "sleep"
    assert window["event_count"] == 1
    assert window["phase_latency_s"] == {"allocator_sleep": 0.2}


def test_write_sleep_profile_summary_csv_flattens_nested_profiles(tmp_path):
    out = tmp_path / "sleep_profile_summary.csv"
    rows = [
        {
            "system": "vllm",
            "run_id": "r1",
            "method": "sleep_l1",
            "model": "m",
            "prompt_name": "short_short",
            "repeat_index": 0,
            "evict": {
                "sleep_profile": {
                    "operation": "sleep",
                    "events": [
                        {
                            "phase": "allocator_sleep",
                            "pid": 123,
                            "latency_s": 0.4,
                            "copy_d2h_s": 0.1,
                            "cpu_backup_alloc_s": 0.25,
                            "accounted_s": 0.39,
                            "unaccounted_s": 0.01,
                            "bytes_by_tag": {"weights": 1024},
                        }
                    ],
                }
            },
        }
    ]

    bench.write_sleep_profile_summary_csv(out, rows)
    text = out.read_text(encoding="utf-8")

    assert "operation,phase" in text
    assert "sleep,allocator_sleep" in text
    assert "cpu_backup_alloc_s" in text
    assert "0.25" in text
    assert '""weights"": 1024' in text
