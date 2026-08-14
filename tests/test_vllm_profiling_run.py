#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
from pathlib import Path

from vllm_switch_bench.experiments.vllm_profiling import run as bench

ROOT = Path(__file__).resolve().parents[1]


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


def test_parse_args_default_out_dir_is_ignored_profiling_staging():
    args = bench.parse_args(["--model", "dummy"])
    assert args.out_dir == "results/tmp/vllm-profiling"


def test_parse_args_accepts_repeatable_extra_vllm_args():
    args = bench.parse_args(
        [
            "--model",
            "dummy",
            "--extra-vllm-arg=--skip-mm-profiling",
            "--extra-vllm-arg=--cpu-offload-gb 2",
        ]
    )
    assert args.extra_vllm_arg == ["--skip-mm-profiling", "--cpu-offload-gb 2"]


def test_parse_args_rejects_non_positive_repeats():
    try:
        bench.parse_args(["--model", "dummy", "--repeats", "0"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("parse_args should reject non-positive repeats")


def test_start_vllm_preserves_virtualenv_bin_on_path(tmp_path, monkeypatch):
    base_python = tmp_path / "base" / "python3"
    base_python.parent.mkdir()
    base_python.touch()
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    venv_python = venv_bin / "python"
    venv_python.symlink_to(base_python)
    args = bench.parse_args(
        [
            "--model",
            "dummy",
            "--python",
            str(venv_python),
            "--workdir",
            str(tmp_path),
        ]
    )
    args.enable_sleep_mode = False
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        kwargs["stdout"].close()
        return object()

    monkeypatch.setattr(bench.subprocess, "Popen", fake_popen)

    bench.start_vllm(args, tmp_path / "server.log")

    assert captured["command"][0] == str(venv_python)
    assert str(venv_bin) in captured["env"]["PATH"].split(os.pathsep)


def test_start_vllm_makes_relative_python_path_absolute(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_HOME", raising=False)
    args = bench.parse_args(
        [
            "--model",
            "dummy",
            "--python",
            "venv/bin/python",
            "--workdir",
            str(tmp_path),
        ]
    )
    args.enable_sleep_mode = False
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        kwargs["stdout"].close()
        return object()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bench.subprocess, "Popen", fake_popen)

    bench.start_vllm(args, tmp_path / "server.log")

    expected_python = tmp_path / "venv" / "bin" / "python"
    assert captured["command"][0] == str(expected_python)
    assert captured["env"]["PATH"].split(os.pathsep)[0] == str(expected_python.parent)


def test_dry_run_metadata_preserves_virtualenv_python_path(tmp_path, monkeypatch):
    base_python = tmp_path / "base" / "python3"
    base_python.parent.mkdir()
    base_python.touch()
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base_python)
    monkeypatch.setattr(
        bench,
        "run_cmd",
        lambda *args, **kwargs: type("CP", (), {"stdout": "0,FakeGPU,10240,999.1\n"})(),
    )

    rc = bench.main(
        [
            "--model",
            "dummy",
            "--python",
            str(venv_python),
            "--workdir",
            str(ROOT),
            "--out-dir",
            str(tmp_path / "results"),
            "--dry-run",
        ]
    )

    assert rc == 0
    created = next(path for path in (tmp_path / "results").iterdir() if path.is_dir())
    metadata = __import__("json").loads((created / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["python"] == str(venv_python)


def test_dry_run_metadata_records_engine_and_benchmark_provenance(tmp_path):
    rc = bench.main(
        [
            "--model",
            "dummy",
            "--workdir",
            str(ROOT),
            "--out-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )
    assert rc == 0
    created = next(path for path in tmp_path.iterdir() if path.is_dir())
    metadata = __import__("json").loads((created / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["benchmark_git"]["commit"]
    assert metadata["engine_git"]["commit"]
    assert isinstance(metadata["benchmark_git"]["dirty"], bool)
    assert isinstance(metadata["engine_git"]["tracked_dirty"], bool)
    assert metadata["experiment"] == "vllm-profiling"
    assert metadata["behavior_config_sha256"]
    assert metadata["model_identity"]["identity"] == "dummy"
    assert metadata["engine_runtime"]["python_path"]


def test_write_summary_csv_flattens_nested_metrics(tmp_path):
    out = tmp_path / "summary.csv"
    bench.write_summary_csv(
        out,
        [
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
                "infer_before": {
                    "ttft_s": 0.1,
                    "client_latency_s": 0.8,
                    "approx_tokens_per_s": 20.0,
                },
                "infer_after": {
                    "ttft_s": 0.2,
                    "client_latency_s": 0.9,
                    "approx_tokens_per_s": 18.0,
                },
            }
        ],
    )
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


def test_write_summary_csv_uses_new_vllm_metric_fields(tmp_path):
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
                "startup_latency_s": 1.5,
                "memory_gpu_used_ready_mib": 1000,
                "memory_gpu_used_evict_mib": 500,
                "evict": {"latency_s": 0.25},
                "restore": {"latency_s": 0.75},
                "infer_before": {"ttft_s": 0.1, "client_latency_s": 0.8, "completion_tokens": 8},
                "infer_after": {"ttft_s": 0.2, "client_latency_s": 0.9, "completion_tokens": 9},
            }
        ],
    )
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["startup_latency_s"] == "1.5"
    assert rows[0]["memory_gpu_used_ready_mib"] == "1000"
    assert rows[0]["memory_gpu_used_evict_mib"] == "500"
    assert rows[0]["output_tokens_before"] == "8"
    assert "startup_to_health_s" not in rows[0]
