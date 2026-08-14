from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_switch_bench.experiments.vllm_profiling.compile import compile_profiles
from vllm_switch_bench.promotion import parse_args


def test_family_help_includes_family_specific_inputs(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        parse_args(["backup-reuse-reclaim", "--help"])

    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--reuse" in help_text
    assert "--reclaim" in help_text


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _repo(path: str, commit: str, *, module_path: str | None = None) -> dict[str, object]:
    return {
        "path": path,
        "commit": commit,
        "branch": "test",
        "dirty": False,
        "tracked_dirty": False,
        "tree": commit,
        "module_path": module_path,
    }


def _service_source(tmp_path: Path, name: str, methods: list[str]) -> Path:
    root = tmp_path / name
    engine = _repo("/runtime/vllm", "e" * 40)
    metadata = {
        "benchmark_git": _repo("/benchmark", "b" * 40),
        "engine_git": engine,
        "engine_runtime": {
            "vllm_import_path": "/runtime/vllm/vllm/__init__.py",
            "vllm_version": "test",
            "python_version": "3.12",
        },
        "model_identity": {"identity": "model", "config_sha256": "c" * 64},
        "behavior_config": {"methods": methods, "enforce_eager": True},
        "behavior_config_sha256": "d" * 64,
        "gpu": "test-gpu",
    }
    _write_json(root / "metadata.json", metadata)
    rows = []
    for method in methods:
        for index in range(6):
            restore: dict[str, object] = {"latency_s": 1.0 + index / 100}
            if method == "sleep_l1":
                restore["sleep_profile"] = {
                    "events": [
                        {
                            "phase": "allocator_wake_up",
                            "create_map_s": 0.4,
                            "copy_h2d_s": 0.5,
                        }
                    ]
                }
            elif method == "sleep_l2":
                restore["steps"] = {
                    "wake_weights": {"latency_s": 0.2},
                    "reload_weights": {"latency_s": 0.5},
                    "wake_kv_cache": {"latency_s": 0.2},
                }
            rows.append({"method": method, "ok": True, "repeat_index": index, "restore": restore})
    summary = root / "summary.json"
    _write_json(summary, rows)
    return summary


def _cpu_source(tmp_path: Path) -> Path:
    path = tmp_path / "cpu" / "summary.json"
    data = {
        "ok": True,
        "environment": {
            "benchmark_repo": _repo("/benchmark", "b" * 40),
            "vllm_repo": _repo(
                "/runtime/vllm-switch",
                "p" * 40,
                module_path="/runtime/vllm-switch/vllm/__init__.py",
            ),
            "python": "3.12",
            "python_executable": "/runtime/python",
            "platform": "test-platform",
            "gpu": "test-gpu",
        },
        "models": [{"name": "model", "path": "/models/model"}],
        "parameters": {"enforce_eager": True},
        "steps": [
            {
                "iteration": index,
                "wake_latency_s": 1.0,
                "wake_allocator_create_map_s": 0.4,
                "wake_allocator_copy_h2d_s": 0.5,
            }
            for index in range(6)
        ],
    }
    _write_json(path, data)
    return path


def _exact_source(tmp_path: Path) -> Path:
    root = tmp_path / "exact"
    raw = root / "raw"
    _write_json(
        raw / "run.json",
        {
            "command_return_code": 0,
            "model": {"name": "model", "path": "/models/model"},
            "environment": {
                "benchmark_repo": _repo("/benchmark", "b" * 40),
                "vllm_repo": _repo("/runtime/vllm-switch", "p" * 40),
                "runtime": {
                    "vllm_import_path": "/runtime/vllm-switch/vllm/__init__.py",
                    "model_config_sha256": "c" * 64,
                },
                "platform": "test-platform",
            },
        },
    )
    _write_json(
        raw / "output_observation.json",
        {"cycles": [{"cycle_index": index, "wake_latency_s": 1.0} for index in range(6)]},
    )
    profile_rows = []
    for _ in range(6):
        profile_rows.extend(
            [
                {"phase": "exact_disk_restore", "disk_pipeline_wall_s": 0.5},
                {
                    "phase": "allocator_wake_up",
                    "create_map_s": 0.4,
                    "disk_restored_bytes_by_tag": {"weights": 1},
                },
            ]
        )
    (raw / "exact_disk_profile.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in profile_rows), encoding="utf-8"
    )
    return root


def test_compile_profiles_accepts_a_complete_local_campaign(tmp_path: Path) -> None:
    compiled = compile_profiles(
        _service_source(tmp_path, "cold", ["cold_reload"]),
        _service_source(tmp_path, "vllm", ["sleep_l1", "sleep_l2"]),
        _cpu_source(tmp_path),
        _exact_source(tmp_path),
    )

    assert len(compiled["samples"]) == 25
    assert set(compiled["source_provenance"]) == set(compiled["sources"])
    assert {sample["method"] for sample in compiled["samples"]} == {
        "Cold load",
        "vLLM L1",
        "vLLM L2",
        "CPU backup",
        "Exact disk",
    }
