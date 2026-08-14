"""Compile validated live-run outputs into the retained profiling sample schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vllm_switch_bench.experiments.vllm_profiling.plot import aggregate_profiles

PHASE_SEMANTICS = {
    "Process + engine startup": "Fresh process creation through API health readiness.",
    "GPU remap": "Allocator mapping/create time.",
    "CPU→GPU copy": "Synchronous pinned CPU backup to GPU copy.",
    "Checkpoint load": "vLLM L2 reload_weights HTTP step.",
    "KV-cache remap": "vLLM L2 wake of the KV-cache tag.",
    "Disk read + hash + H2D pipeline": "Overlapped exact-disk restore pipeline wall time.",
    "Control overhead": "Measured total minus the disjoint instrumented phases.",
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_identity(data: dict[str, Any]) -> dict[str, Any]:
    """Keep stable repository identity while excluding verbose status listings."""

    return {
        "path": data.get("path", data.get("repo_path")),
        "commit": data.get("commit", data.get("git_commit")),
        "branch": data.get("branch", data.get("git_branch")),
        "dirty": data.get("dirty", data.get("git_dirty")),
        "tracked_dirty": data.get("tracked_dirty", data.get("git_tracked_dirty")),
        "tree": data.get("tree"),
        "module_path": data.get("module_path"),
    }


def _service_provenance(summary: Path) -> dict[str, Any]:
    metadata = _json(summary.with_name("metadata.json"))
    return {
        "kind": "fresh-process-vllm-service",
        "benchmark_repo": _git_identity(metadata["benchmark_git"]),
        "engine_repo": _git_identity(metadata["engine_git"]),
        "engine_runtime": metadata["engine_runtime"],
        "model_identity": metadata["model_identity"],
        "behavior_config": metadata["behavior_config"],
        "behavior_config_sha256": metadata["behavior_config_sha256"],
        "gpu": str(metadata["gpu"]).strip(),
    }


def _cpu_provenance(summary: Path) -> dict[str, Any]:
    data = _json(summary)
    environment = data["environment"]
    return {
        "kind": "same-process-vllm-switch-cpu-backup",
        "benchmark_repo": _git_identity(environment["benchmark_repo"]),
        "engine_repo": _git_identity(environment["vllm_repo"]),
        "runtime": {
            "python": environment["python"],
            "python_executable": environment["python_executable"],
            "platform": environment["platform"],
        },
        "model_identity": data["models"],
        "behavior_config": data["parameters"],
        "gpu": environment["gpu"],
    }


def _exact_provenance(run: Path) -> dict[str, Any]:
    data = _json(run / "raw" / "run.json")
    environment = data["environment"]
    return {
        "kind": "same-process-vllm-switch-exact-disk",
        "benchmark_repo": _git_identity(environment["benchmark_repo"]),
        "engine_repo": _git_identity(environment["vllm_repo"]),
        "runtime": environment["runtime"],
        "model_identity": data["model"],
        "command_return_code": data["command_return_code"],
        "platform": environment["platform"],
    }


def _rows(path: Path, method: str) -> list[dict[str, Any]]:
    rows = [row for row in _json(path) if row.get("method") == method and row.get("ok") is True]
    retained = [row for row in rows if int(row["repeat_index"]) > 0]
    if len(retained) != 5:
        raise ValueError(f"{method}: expected five successful post-warm-up samples")
    return sorted(retained, key=lambda row: int(row["repeat_index"]))


def _residual(total: float, phases: dict[str, float]) -> dict[str, float]:
    residual = total - sum(phases.values())
    if residual < -1e-6:
        raise ValueError(f"instrumented phases exceed measured total by {-residual} seconds")
    phases["Control overhead"] = max(residual, 0.0)
    return phases


def compile_profiles(
    cold_summary: Path,
    vllm_summary: Path,
    cpu_summary: Path,
    exact_run: Path,
) -> dict[str, Any]:
    sources = {
        "cold": "cold-run",
        "vllm": "vllm-profile-run",
        "cpu": "vllm-switch-cpu-run",
        "exact": "vllm-switch-exact-disk-run",
    }
    samples: list[dict[str, Any]] = []
    for row in _rows(cold_summary, "cold_reload"):
        total = float(row["restore"]["latency_s"])
        samples.append(
            {
                "method": "Cold load",
                "sample_index": int(row["repeat_index"]),
                "total_s": total,
                "phases_s": {"Process + engine startup": total},
                "source": sources["cold"],
            }
        )

    for row in _rows(vllm_summary, "sleep_l1"):
        total = float(row["restore"]["latency_s"])
        events = row["restore"]["sleep_profile"]["events"]
        allocator = next(item for item in events if item.get("phase") == "allocator_wake_up")
        phases = {
            "GPU remap": float(allocator["create_map_s"]),
            "CPU→GPU copy": float(allocator["copy_h2d_s"]),
        }
        samples.append(
            {
                "method": "vLLM L1",
                "sample_index": int(row["repeat_index"]),
                "total_s": total,
                "phases_s": _residual(total, phases),
                "source": sources["vllm"],
            }
        )

    for row in _rows(vllm_summary, "sleep_l2"):
        total = float(row["restore"]["latency_s"])
        steps = row["restore"]["steps"]
        phases = {
            "GPU remap": float(steps["wake_weights"]["latency_s"]),
            "Checkpoint load": float(steps["reload_weights"]["latency_s"]),
            "KV-cache remap": float(steps["wake_kv_cache"]["latency_s"]),
        }
        samples.append(
            {
                "method": "vLLM L2",
                "sample_index": int(row["repeat_index"]),
                "total_s": total,
                "phases_s": _residual(total, phases),
                "source": sources["vllm"],
            }
        )

    cpu = _json(cpu_summary)
    if cpu.get("ok") is not True:
        raise ValueError("CPU backup source run did not pass its assertions")
    cpu_steps = [step for step in cpu["steps"] if int(step["iteration"]) > 0]
    if len(cpu_steps) != 5:
        raise ValueError("CPU backup: expected five post-warm-up wake samples")
    for step in cpu_steps:
        total = float(step["wake_latency_s"])
        phases = {
            "GPU remap": float(step["wake_allocator_create_map_s"]),
            "CPU→GPU copy": float(step["wake_allocator_copy_h2d_s"]),
        }
        samples.append(
            {
                "method": "CPU backup",
                "sample_index": int(step["iteration"]),
                "total_s": total,
                "phases_s": _residual(total, phases),
                "source": sources["cpu"],
            }
        )

    output_path = exact_run / "raw" / "output_observation.json"
    profile_path = exact_run / "raw" / "exact_disk_profile.jsonl"
    output = _json(output_path)
    cycles = [cycle for cycle in output["cycles"] if int(cycle["cycle_index"]) > 0]
    restores = [
        json.loads(line)
        for line in profile_path.read_text(encoding="utf-8").splitlines()
        if line and json.loads(line).get("phase") == "exact_disk_restore"
    ][-5:]
    wakes = [
        json.loads(line)
        for line in profile_path.read_text(encoding="utf-8").splitlines()
        if line
        and json.loads(line).get("phase") == "allocator_wake_up"
        and json.loads(line).get("disk_restored_bytes_by_tag")
    ][-5:]
    if len(cycles) != len(restores) or len(cycles) != len(wakes) or len(cycles) != 5:
        raise ValueError("exact disk: expected five post-warm-up cycles and profile pairs")
    for cycle, restore, wake in zip(cycles, restores, wakes, strict=True):
        total = float(cycle["wake_latency_s"])
        phases = {
            "Disk read + hash + H2D pipeline": float(restore["disk_pipeline_wall_s"]),
            "GPU remap": float(wake["create_map_s"]),
        }
        samples.append(
            {
                "method": "Exact disk",
                "sample_index": int(cycle["cycle_index"]),
                "total_s": total,
                "phases_s": _residual(total, phases),
                "source": sources["exact"],
            }
        )

    document = {
        "schema_version": 1,
        "title": "Qwen2.5-0.5B activation latency profiling",
        "metric_boundary": (
            "Activation begins immediately before process start or wake API call and ends "
            "when the health/wake response returns; request generation is excluded."
        ),
        "model": "Qwen2.5-0.5B-Instruct",
        "frozen_scope": {
            "gpu": "NVIDIA GeForce RTX 3080 10 GiB",
            "dtype": "float16",
            "max_model_len": 1024,
            "gpu_memory_utilization": 0.8,
            "engine_mode": "eager",
            "sample_count_per_method": 5,
        },
        "stability_rule": {
            "warmup": "Discard sample/cycle index 0 for every method.",
            "center": "median of five retained samples",
            "spread": "minimum and maximum of five retained samples",
            "profile": "real sample nearest the median; ties use sample_index",
        },
        "phase_semantics": PHASE_SEMANTICS,
        "sources": list(sources.values()),
        "source_provenance": {
            sources["cold"]: _service_provenance(cold_summary),
            sources["vllm"]: _service_provenance(vllm_summary),
            sources["cpu"]: _cpu_provenance(cpu_summary),
            sources["exact"]: _exact_provenance(exact_run),
        },
        "samples": samples,
        "evidence_label": "descriptive local mechanism comparison",
        "comparability": {
            "shared_conditions": [
                "Qwen2.5-0.5B-Instruct",
                "NVIDIA RTX 3080",
                "float16",
                "max_model_len=1024",
                "gpu_memory_utilization=0.80",
                "eager execution",
            ],
            "heterogeneous_conditions": [
                "Cold/L1/L2 launch fresh service processes; CPU and disk use same-process cycles."
            ],
            "prohibited_claim": "Do not infer a release-matched system ranking.",
        },
    }
    aggregate_profiles(document)
    return document
