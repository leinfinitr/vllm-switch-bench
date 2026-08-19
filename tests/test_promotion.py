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
            "torch_version": "2.11.0",
            "torch_cuda_version": "13.0",
        },
        "model_identity": {"identity": "model", "config_sha256": "c" * 64},
        "behavior_config": {"methods": methods, "enforce_eager": True},
        "behavior_config_sha256": "d" * 64,
        "gpu": "test-gpu",
    }
    _write_json(root / "metadata.json", metadata)
    rows = []
    for method in methods:
        for index in range(3):
            evict: dict[str, object] = {"latency_s": 0.5 + index / 100}
            restore: dict[str, object] = {"latency_s": 1.0 + index / 100}
            if method in {"sleep_l1", "sleep_l2"}:
                evict["sleep_profile"] = {
                    "events": [
                        {
                            "phase": "allocator_sleep",
                            "cpu_backup_alloc_s": 0.1 if method == "sleep_l1" else 0.0,
                            "copy_d2h_s": 0.1 if method == "sleep_l1" else 0.0,
                            "unmap_release_s": 0.2,
                        }
                    ]
                }
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
            rows.append(
                {
                    "method": method,
                    "ok": True,
                    "repeat_index": index,
                    "evict": evict,
                    "restore": restore,
                }
            )
    summary = root / "summary.json"
    _write_json(summary, rows)
    return summary


def _block_source(tmp_path: Path, name: str, methods: list[str]) -> Path:
    root = tmp_path / name
    for method in methods:
        for block in range(3):
            environment = {
                "benchmark_repo": _repo("/benchmark", "b" * 40),
                "vllm_repo": _repo(
                    f"/runtime/{name}",
                    "p" * 40,
                    module_path=f"/runtime/{name}/vllm/__init__.py",
                ),
                "runtime": {
                    "python_version": "3.12",
                    "torch_version": "2.11.0",
                    "torch_cuda_version": "13.0",
                    "vllm_import_path": f"/runtime/{name}/vllm/__init__.py",
                },
                "platform": "test-platform",
            }
            cycles = []
            cold_index = block
            for cycle in range(3):
                allocator_sleep = {
                    "phase": "allocator_sleep",
                    "cpu_backup_alloc_s": 0.1,
                    "copy_d2h_s": 0.1,
                    "unmap_release_s": 0.2,
                }
                allocator_wake = {
                    "phase": "allocator_wake_up",
                    "create_map_s": 0.4,
                    "copy_h2d_s": 0.5,
                }
                restore: dict[str, object] = {
                    "latency_s": 1.0,
                    "sleep_profile": {"events": [allocator_wake]},
                }
                cache = None
                condition = None
                if method == "sleep_l2":
                    condition = "cold" if cycle == cold_index else "warm"
                    restore = {
                        "latency_s": 1.0,
                        "steps": {
                            "wake_weights": {"latency_s": 0.2},
                            "reload_weights": {"latency_s": 0.5},
                            "wake_kv_cache": {"latency_s": 0.2},
                        },
                    }
                    cold = condition == "cold"
                    cache = {
                        "valid": True,
                        "treatment": "posix_fadvise_dontneed" if cold else "none",
                        "before_wake": {
                            "resident_ratio": 0.0 if cold else 1.0,
                        },
                        "io_delta": {
                            "read_bytes": 1000 if cold else 0,
                            "major_faults": 1 if cold else 0,
                        },
                        "storage_read_ratio": 1.0 if cold else 0.0,
                    }
                elif method == "exact_disk":
                    restore["sleep_profile"] = {
                        "events": [
                            allocator_wake,
                            {"phase": "exact_disk_restore", "disk_pipeline_wall_s": 0.5},
                        ]
                    }
                cycles.append(
                    {
                        "cycle_index": cycle,
                        "cycle_class": "first" if cycle == 0 else "steady",
                        "cache_condition": condition,
                        "sleep": {
                            "latency_s": 0.5,
                            "sleep_profile": {"events": [allocator_sleep]},
                        },
                        "restore": restore,
                        "cache_observation": cache,
                        "ok": True,
                    }
                )
            _write_json(
                root / method / f"block-{block}" / "block-summary.json",
                {
                    "ok": True,
                    "method": method,
                    "process_block": block,
                    "model": "/models/model",
                    "environment": environment,
                    "parameters": {"enforce_eager": True},
                    "cycles": cycles,
                },
            )
    return root


def test_compile_profiles_accepts_a_complete_local_campaign(tmp_path: Path) -> None:
    compiled = compile_profiles(
        _service_source(tmp_path, "cold", ["cold_reload"]),
        _block_source(tmp_path, "vllm", ["sleep_l1", "sleep_l2"]),
        _block_source(tmp_path, "switch", ["cpu_backup", "exact_disk"]),
    )

    assert len(compiled["samples"]) == 27
    assert compiled["schema_version"] == 3
    assert all("sleep_phases_s" in sample for sample in compiled["samples"])
    assert all("wake_phases_s" in sample for sample in compiled["samples"])
    assert set(compiled["source_provenance"]) == set(compiled["sources"])
    assert {sample["method"] for sample in compiled["samples"]} == {
        "Cold load",
        "vLLM L1 First",
        "vLLM L1 Steady",
        "vLLM L2 Cold",
        "vLLM L2 Warm",
        "CPU backup First",
        "CPU backup Steady",
        "Exact disk First",
        "Exact disk Steady",
    }


def test_compile_profiles_rejects_method_identity_mismatch(tmp_path: Path) -> None:
    cold = _service_source(tmp_path, "cold", ["cold_reload"])
    vllm = _block_source(tmp_path, "vllm", ["sleep_l1", "sleep_l2"])
    switch = _block_source(tmp_path, "switch", ["cpu_backup", "exact_disk"])
    path = vllm / "sleep_l1" / "block-1" / "block-summary.json"
    block = json.loads(path.read_text())
    block["method"] = "sleep_l2"
    _write_json(path, block)

    with pytest.raises(ValueError, match="method identity mismatch"):
        compile_profiles(cold, vllm, switch)


def test_compile_profiles_rejects_mixed_block_configuration(tmp_path: Path) -> None:
    cold = _service_source(tmp_path, "cold", ["cold_reload"])
    vllm = _block_source(tmp_path, "vllm", ["sleep_l1", "sleep_l2"])
    switch = _block_source(tmp_path, "switch", ["cpu_backup", "exact_disk"])
    path = switch / "cpu_backup" / "block-2" / "block-summary.json"
    block = json.loads(path.read_text())
    block["parameters"]["gpu_memory_utilization"] = 0.5
    _write_json(path, block)

    with pytest.raises(ValueError, match="environment|parameters"):
        compile_profiles(cold, vllm, switch)


def test_compile_profiles_rejects_cross_system_runtime_mismatch(tmp_path: Path) -> None:
    cold = _service_source(tmp_path, "cold", ["cold_reload"])
    vllm = _block_source(tmp_path, "vllm", ["sleep_l1", "sleep_l2"])
    switch = _block_source(tmp_path, "switch", ["cpu_backup", "exact_disk"])
    for path in switch.glob("*/block-*/block-summary.json"):
        block = json.loads(path.read_text())
        block["environment"]["runtime"]["python_version"] = "3.14"
        _write_json(path, block)

    with pytest.raises(ValueError, match="same Python version"):
        compile_profiles(cold, vllm, switch)


def test_compile_profiles_rejects_wrong_l2_cache_schedule(tmp_path: Path) -> None:
    cold = _service_source(tmp_path, "cold", ["cold_reload"])
    vllm = _block_source(tmp_path, "vllm", ["sleep_l1", "sleep_l2"])
    switch = _block_source(tmp_path, "switch", ["cpu_backup", "exact_disk"])
    path = vllm / "sleep_l2" / "block-1" / "block-summary.json"
    block = json.loads(path.read_text())
    block["cycles"][0]["cache_condition"] = "cold"
    block["cycles"][1]["cache_condition"] = "warm"
    _write_json(path, block)

    with pytest.raises(ValueError, match="cache schedule"):
        compile_profiles(cold, vllm, switch)
