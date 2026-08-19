from __future__ import annotations

import json

import pytest

from vllm_switch_bench.experiments.vllm_profiling import run as bench
from vllm_switch_bench.experiments.vllm_profiling.page_cache import (
    evict_page_cache,
    l2_cache_schedule,
    measure_page_cache,
    validate_cache_observation,
)


def test_vllm_harness_uses_shared_prompt_catalog():
    from vllm_switch_bench.common.schema import PROMPTS

    assert bench.PROMPTS is PROMPTS


def test_vllm_harness_writes_summary_csv(tmp_path):
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
                "infer_before": {
                    "ttft_s": 0.1,
                    "client_latency_s": 0.2,
                    "approx_tokens_per_s": 3.0,
                },
                "infer_after": {"ttft_s": 0.2, "client_latency_s": 0.3, "approx_tokens_per_s": 4.0},
            }
        ],
    )
    text = out.read_text()
    assert text.splitlines()[0].startswith("system,")
    assert ",vllm," in text or text.endswith(",vllm") or "vllm" in text


def test_vllm_main_dry_run_metadata_records_system(tmp_path, monkeypatch):
    monkeypatch.setattr(
        bench,
        "run_cmd",
        lambda *args, **kwargs: type("CP", (), {"stdout": "0,FakeGPU,10240,999.1\n"})(),
    )
    rc = bench.main(
        [
            "--model",
            "dummy",
            "--out-dir",
            str(tmp_path),
            "--sleep-cpu-backup-pin-memory",
            "false",
            "--dry-run",
        ]
    )
    assert rc == 0
    created = [p for p in tmp_path.iterdir() if p.is_dir()]
    metadata = (created[0] / "metadata.json").read_text()
    assert '"system": "vllm"' in metadata
    assert '"sleep_cpu_backup_pin_memory": "false"' in metadata


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


def test_combine_restore_steps_uses_continuous_envelope():
    combined = bench.combine_restore_steps(
        {"ok": True, "status": 200, "latency_s": 0.2},
        {"ok": True, "status": 200, "latency_s": 0.5},
        {"ok": True, "status": 200, "latency_s": 0.2},
        started_monotonic_s=10.0,
        ended_monotonic_s=11.1,
    )

    assert combined["latency_s"] == pytest.approx(1.1)
    assert combined["active_latency_s"] == pytest.approx(0.9)
    assert combined["inter_step_gap_s"] == pytest.approx(0.2)


def test_l2_cache_schedule_rotates_cold_cycle():
    assert l2_cache_schedule(0, 3) == ["cold", "warm", "warm"]
    assert l2_cache_schedule(1, 3) == ["warm", "cold", "warm"]
    assert l2_cache_schedule(2, 3) == ["warm", "warm", "cold"]


def test_page_cache_measure_and_evict_are_file_scoped(tmp_path):
    payload = tmp_path / "model.safetensors"
    payload.write_bytes(b"x" * (2 * 1024 * 1024))
    payload.read_bytes()

    before = measure_page_cache([payload])
    treatment = evict_page_cache([payload])

    assert before["total_bytes"] == payload.stat().st_size
    assert treatment["ok"] is True
    assert treatment["after"]["resident_ratio"] <= treatment["before"]["resident_ratio"]


@pytest.mark.parametrize(
    ("condition", "resident", "read_bytes", "valid"),
    [
        ("cold", 0.01, 950, True),
        ("cold", 0.20, 950, False),
        ("warm", 0.99, 0, True),
        ("warm", 0.99, 500, False),
    ],
)
def test_validate_cache_observation(condition, resident, read_bytes, valid):
    observed, failures = validate_cache_observation(
        condition,
        before_wake={"resident_ratio": resident},
        io_delta={"read_bytes": read_bytes},
        checkpoint_bytes=1000,
        cold_max_resident_ratio=0.05,
        cold_min_read_ratio=0.90,
        warm_min_resident_ratio=0.90,
        warm_max_read_ratio=0.10,
    )

    assert observed is valid
    assert bool(failures) is (not valid)
