"""Compile local process-block outputs into compact retained profiling samples."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from vllm_switch_bench.experiments.vllm_profiling.plot import aggregate_profiles

PHASE_SEMANTICS = {
    "Process shutdown": "Process termination request through process exit.",
    "CPU backup allocation": "Pinned CPU backup allocation during sleep.",
    "GPU→CPU copy": "Synchronous GPU weights backup to CPU memory.",
    "GPU unmap + release": "Allocator unmap and GPU allocation release during sleep.",
    "Process + engine startup": "Fresh process creation through API health readiness.",
    "GPU remap": "Allocator mapping/create time across all restored tags.",
    "CPU→GPU copy": "Synchronous pinned CPU backup to GPU copy.",
    "Checkpoint load": "vLLM L2 reload_weights active step.",
    "KV-cache remap": "vLLM L2 wake of the KV-cache tag.",
    "Disk read + hash + H2D pipeline": "Overlapped exact-disk restore pipeline wall time.",
    "Control overhead": "Continuous operation wall time minus disjoint instrumented phases.",
}
METRIC_BOUNDARY = {
    "sleep": (
        "Sleep begins immediately before process termination or the sleep call and ends when "
        "process exit or sleep returns."
    ),
    "wake": (
        "Wake begins immediately before process start or the first restore call and ends after "
        "all required restore stages return; request generation and cache treatment are excluded."
    ),
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_identity(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": data.get("path", data.get("repo_path")),
        "commit": data.get("commit", data.get("git_commit")),
        "branch": data.get("branch", data.get("git_branch")),
        "dirty": data.get("dirty", data.get("git_dirty")),
        "tracked_dirty": data.get("tracked_dirty", data.get("git_tracked_dirty")),
        "tree": data.get("tree"),
        "working_tree_sha256": data.get("working_tree_sha256"),
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


def _block_files(root: Path, method: str) -> list[Path]:
    files = sorted((root / method).glob("block-*/block-summary.json"))
    if not files:
        if root.name == method and (root / "block-summary.json").is_file():
            files = [root / "block-summary.json"]
        elif (root / "block-summary.json").is_file():
            files = [root / "block-summary.json"]
    return files


def _blocks(root: Path, method: str, expected: int = 3) -> list[dict[str, Any]]:
    rows = [_json(path) for path in _block_files(root, method)]
    rows = sorted(rows, key=lambda row: int(row["process_block"]))
    if len(rows) != expected:
        raise ValueError(f"{method}: expected {expected} process blocks, found {len(rows)}")
    if [int(row["process_block"]) for row in rows] != list(range(expected)):
        raise ValueError(f"{method}: process block indexes must be 0..{expected - 1}")
    if not all(row.get("method") == method for row in rows):
        raise ValueError(f"{method}: source block method identity mismatch")
    if not all(row.get("ok") is True for row in rows):
        raise ValueError(f"{method}: source blocks include failed or invalid cycles")
    if not all(len(row.get("cycles", [])) == 3 for row in rows):
        raise ValueError(f"{method}: every block must contain exactly three cycles")

    expected_cycle_classes = ["first", "steady", "steady"]
    expected_cache_conditions = {
        "sleep_l1": [None, None, None],
        "sleep_l2": None,
        "cpu_backup": [None, None, None],
        "exact_disk": [None, None, None],
    }
    for row in rows:
        cycles = row["cycles"]
        if [int(cycle.get("cycle_index", -1)) for cycle in cycles] != [0, 1, 2]:
            raise ValueError(f"{method}: cycle indexes must be 0, 1, 2")
        if [cycle.get("cycle_class") for cycle in cycles] != expected_cycle_classes:
            raise ValueError(f"{method}: cycle classes must be first, steady, steady")
        if not all(cycle.get("ok") is True for cycle in cycles):
            raise ValueError(f"{method}: every cycle must succeed")
        conditions = [cycle.get("cache_condition") for cycle in cycles]
        if method == "sleep_l2":
            expected_conditions = ["warm", "warm", "warm"]
            expected_conditions[int(row["process_block"]) % 3] = "cold"
            if conditions != expected_conditions:
                raise ValueError(
                    f"{method}: block {row['process_block']} cache schedule must be "
                    f"{expected_conditions}, found {conditions}"
                )
        elif conditions != expected_cache_conditions[method]:
            raise ValueError(f"{method}: only L2 cycles may carry cache conditions")
    return rows


def _same_across_blocks(blocks: list[dict[str, Any]], path: str) -> Any:
    keys = path.split(".")
    values = []
    for block in blocks:
        value: Any = block
        for key in keys:
            value = value[key]
        values.append(value)
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"process blocks disagree on {path}")
    return values[0]


def _block_provenance(blocks: list[dict[str, Any]], *, kind: str) -> dict[str, Any]:
    benchmark_repo = _same_across_blocks(blocks, "environment.benchmark_repo")
    engine_repo = _same_across_blocks(blocks, "environment.vllm_repo")
    runtime = _same_across_blocks(blocks, "environment.runtime")
    model = _same_across_blocks(blocks, "model")
    parameters = _same_across_blocks(blocks, "parameters")
    return {
        "kind": kind,
        "benchmark_repo": _git_identity(benchmark_repo),
        "engine_repo": _git_identity(engine_repo),
        "engine_runtime": runtime,
        "model_identity": {"identity": model},
        "behavior_config": parameters,
        "process_block_count": len(blocks),
        "cycles_per_process": len(blocks[0]["cycles"]),
    }


def _residual(total: float, phases: dict[str, float]) -> dict[str, float]:
    residual = total - sum(phases.values())
    if residual < -1e-6:
        raise ValueError(f"instrumented phases exceed measured total by {-residual} seconds")
    phases["Control overhead"] = max(residual, 0.0)
    return phases


def _single_phase_event(profile: dict[str, Any], phase: str) -> dict[str, Any]:
    matching = [event for event in profile["events"] if event.get("phase") == phase]
    if len(matching) != 1:
        raise ValueError(f"expected exactly one {phase!r} event, found {len(matching)}")
    return matching[0]


def _allocator_event(profile: dict[str, Any], phase: str) -> dict[str, Any]:
    return _single_phase_event(profile, phase)


def _sleep_phases(cycle: dict[str, Any]) -> dict[str, float]:
    event = _allocator_event(cycle["sleep"]["sleep_profile"], "allocator_sleep")
    return {
        "CPU backup allocation": float(event["cpu_backup_alloc_s"]),
        "GPU→CPU copy": float(event["copy_d2h_s"]),
        "GPU unmap + release": float(event["unmap_release_s"]),
    }


def _wake_phases(cycle: dict[str, Any], method: str) -> dict[str, float]:
    if method == "vLLM L2 Cold" or method == "vLLM L2 Warm":
        steps = cycle["restore"]["steps"]
        weight_event = _allocator_event(steps["wake_weights"]["sleep_profile"], "allocator_wake_up")
        reload_event = _single_phase_event(
            steps["reload_weights"]["sleep_profile"], "reload_weights"
        )
        kv_event = _allocator_event(steps["wake_kv_cache"]["sleep_profile"], "allocator_wake_up")
        return {
            "GPU remap": float(weight_event["create_map_s"]),
            "Checkpoint load": float(reload_event["latency_s"]),
            "KV-cache remap": float(kv_event["create_map_s"]),
        }
    event = _allocator_event(cycle["restore"]["sleep_profile"], "allocator_wake_up")
    phases = {"GPU remap": float(event["create_map_s"])}
    if not method.startswith("Exact disk"):
        phases["CPU→GPU copy"] = float(event["copy_h2d_s"])
        return phases
    restore_event = _single_phase_event(cycle["restore"]["sleep_profile"], "exact_disk_restore")
    disk_bytes = int(event.get("disk_restored_bytes_by_tag", {}).get("weights", 0))
    if (
        restore_event.get("source_medium") != "disk"
        or restore_event.get("fallback") is not False
        or int(restore_event.get("disk_read_bytes", 0)) <= 0
        or disk_bytes <= 0
        or int(event.get("cpu_restored_bytes_by_tag", {}).get("weights", 0)) != 0
        or float(event.get("copy_h2d_s", 0.0)) != 0.0
    ):
        raise ValueError("exact disk: restore mechanism evidence is invalid")
    phases["Disk read + hash + H2D pipeline"] = float(restore_event["disk_pipeline_wall_s"])
    return phases


def _compact_sample(
    *,
    method: str,
    block: dict[str, Any],
    cycles: list[dict[str, Any]],
    source: str,
) -> dict[str, Any]:
    if not cycles:
        raise ValueError(f"{method}: block observation has no cycles")
    sleep_total = statistics.mean(float(cycle["sleep"]["latency_s"]) for cycle in cycles)
    wake_total = statistics.mean(float(cycle["restore"]["latency_s"]) for cycle in cycles)

    def mean_phases(operation: str) -> dict[str, float]:
        breakdowns = [
            _sleep_phases(cycle) if operation == "sleep" else _wake_phases(cycle, method)
            for cycle in cycles
        ]
        names = set().union(*(breakdown.keys() for breakdown in breakdowns))
        return {
            name: statistics.mean(float(breakdown.get(name, 0.0)) for breakdown in breakdowns)
            for name in names
        }

    row = {
        "method": method,
        "sample_index": int(block["process_block"]) + 1,
        "process_block": int(block["process_block"]),
        "cycle_indices": [int(cycle["cycle_index"]) for cycle in cycles],
        "cycle_class": cycles[0]["cycle_class"],
        "cache_condition": cycles[0].get("cache_condition"),
        "sleep_total_s": sleep_total,
        "sleep_phases_s": _residual(sleep_total, mean_phases("sleep")),
        "wake_total_s": wake_total,
        "wake_phases_s": _residual(wake_total, mean_phases("wake")),
        "source": source,
    }
    if method in {"vLLM L2 Cold", "vLLM L2 Warm"}:
        active = statistics.mean(float(cycle["restore"]["active_latency_s"]) for cycle in cycles)
        gap = statistics.mean(float(cycle["restore"]["inter_step_gap_s"]) for cycle in cycles)
        if abs(wake_total - active - gap) > 1e-6:
            raise ValueError(f"{method}: continuous wake envelope does not close")
        row["wake_active_s"] = active
        row["wake_inter_step_gap_s"] = gap
    if method.startswith("Exact disk"):
        allocator_events = [
            _allocator_event(cycle["restore"]["sleep_profile"], "allocator_wake_up")
            for cycle in cycles
        ]
        restore_events = [
            _single_phase_event(cycle["restore"]["sleep_profile"], "exact_disk_restore")
            for cycle in cycles
        ]
        row["mechanism_evidence"] = {
            "source_medium": "disk",
            "fallback": False,
            "disk_read_bytes": statistics.mean(
                int(event["disk_read_bytes"]) for event in restore_events
            ),
            "disk_restored_weight_bytes": statistics.mean(
                int(event["disk_restored_bytes_by_tag"]["weights"]) for event in allocator_events
            ),
            "cpu_restored_weight_bytes": statistics.mean(
                int(event.get("cpu_restored_bytes_by_tag", {}).get("weights", 0))
                for event in allocator_events
            ),
        }
    caches = [cycle.get("cache_observation") for cycle in cycles]
    if any(cache is not None for cache in caches):
        if any(cache is None or cache.get("valid") is not True for cache in caches):
            failures = [None if cache is None else cache.get("failures") for cache in caches]
            raise ValueError(f"{method}: cache-state validation failed: {failures}")
        valid_caches: list[dict[str, Any]] = [cache for cache in caches if cache is not None]
        row["cache_evidence"] = {
            "treatment": valid_caches[0]["treatment"],
            "resident_ratio_before_wake": statistics.mean(
                float(cache["before_wake"]["resident_ratio"]) for cache in valid_caches
            ),
            "storage_read_bytes": statistics.mean(
                int(cache["io_delta"]["read_bytes"]) for cache in valid_caches
            ),
            "storage_read_ratio": statistics.mean(
                float(cache["storage_read_ratio"]) for cache in valid_caches
            ),
            "major_faults": statistics.mean(
                int(cache["io_delta"]["major_faults"]) for cache in valid_caches
            ),
        }
    return row


def _steady_cycles(block: dict[str, Any]) -> list[dict[str, Any]]:
    return [cycle for cycle in block["cycles"] if cycle["cycle_class"] == "steady"]


def _service_rows(path: Path, method: str, expected: int) -> list[dict[str, Any]]:
    rows = [row for row in _json(path) if row.get("method") == method and row.get("ok") is True]
    if len(rows) != expected:
        raise ValueError(f"{method}: expected {expected} successful samples")
    return sorted(rows, key=lambda row: int(row["repeat_index"]))


def _compile_cold(path: Path, source: str) -> list[dict[str, Any]]:
    samples = []
    for row in _service_rows(path, "cold_reload", 3):
        sleep_total = float(row["evict"]["latency_s"])
        wake_total = float(row["restore"]["latency_s"])
        samples.append(
            {
                "method": "Cold load",
                "sample_index": int(row["repeat_index"]) + 1,
                "process_block": int(row["repeat_index"]),
                "cycle_index": None,
                "cycle_class": "cold_process",
                "cache_condition": None,
                "sleep_total_s": sleep_total,
                "sleep_phases_s": {"Process shutdown": sleep_total},
                "wake_total_s": wake_total,
                "wake_phases_s": {"Process + engine startup": wake_total},
                "source": source,
            }
        )
    return samples


def compile_profiles(
    cold_summary: Path,
    vllm_blocks: Path,
    switch_blocks: Path,
) -> dict[str, Any]:
    sources = {
        "cold": "cold-run",
        "vllm": "vllm-process-blocks",
        "switch": "vllm-switch-process-blocks",
    }
    samples = _compile_cold(cold_summary, sources["cold"])
    l1_blocks = _blocks(vllm_blocks, "sleep_l1")
    l2_blocks = _blocks(vllm_blocks, "sleep_l2")
    cpu_blocks = _blocks(switch_blocks, "cpu_backup")
    disk_blocks = _blocks(switch_blocks, "exact_disk")

    for block in l1_blocks:
        samples.append(
            _compact_sample(
                method="vLLM L1 First",
                block=block,
                cycles=[block["cycles"][0]],
                source=sources["vllm"],
            )
        )
        samples.append(
            _compact_sample(
                method="vLLM L1 Steady",
                block=block,
                cycles=_steady_cycles(block),
                source=sources["vllm"],
            )
        )
    for condition, method in (("cold", "vLLM L2 Cold"), ("warm", "vLLM L2 Warm")):
        for block in l2_blocks:
            matching = [cycle for cycle in block["cycles"] if cycle["cache_condition"] == condition]
            samples.append(
                _compact_sample(
                    method=method,
                    block=block,
                    cycles=matching,
                    source=sources["vllm"],
                )
            )
    for blocks, first_method, steady_method in (
        (cpu_blocks, "CPU backup First", "CPU backup Steady"),
        (disk_blocks, "Exact disk First", "Exact disk Steady"),
    ):
        for block in blocks:
            samples.append(
                _compact_sample(
                    method=first_method,
                    block=block,
                    cycles=[block["cycles"][0]],
                    source=sources["switch"],
                )
            )
            samples.append(
                _compact_sample(
                    method=steady_method,
                    block=block,
                    cycles=_steady_cycles(block),
                    source=sources["switch"],
                )
            )

    vllm_runtime = l1_blocks[0]["environment"]["runtime"]
    switch_runtime = cpu_blocks[0]["environment"]["runtime"]
    if vllm_runtime.get("python_version") != switch_runtime.get("python_version"):
        raise ValueError("vLLM and vllm-switch must use the same Python version")
    if vllm_runtime.get("torch_version") != switch_runtime.get("torch_version"):
        raise ValueError("vLLM and vllm-switch must use the same Torch version")
    if vllm_runtime.get("torch_cuda_version") != switch_runtime.get("torch_cuda_version"):
        raise ValueError("vLLM and vllm-switch must use the same Torch CUDA version")

    cold_provenance = _service_provenance(cold_summary)
    cold_runtime = cold_provenance["engine_runtime"]
    for field, label in (
        ("python_version", "Python"),
        ("torch_version", "Torch"),
        ("torch_cuda_version", "Torch CUDA"),
    ):
        values = {cold_runtime.get(field), vllm_runtime.get(field), switch_runtime.get(field)}
        if len(values) != 1:
            raise ValueError(f"cold/vLLM/vllm-switch must use the same {label} version")

    source_provenance = {
        sources["cold"]: cold_provenance,
        sources["vllm"]: {
            "kind": "same-process-native-vllm-blocks",
            "methods": {
                "sleep_l1": _block_provenance(l1_blocks, kind="same-process-native-vllm-l1"),
                "sleep_l2": _block_provenance(l2_blocks, kind="same-process-native-vllm-l2"),
            },
            "benchmark_repo": _git_identity(l1_blocks[0]["environment"]["benchmark_repo"]),
            "engine_repo": _git_identity(l1_blocks[0]["environment"]["vllm_repo"]),
            "engine_runtime": l1_blocks[0]["environment"]["runtime"],
            "model_identity": {"identity": l1_blocks[0]["model"]},
        },
        sources["switch"]: {
            "kind": "same-process-vllm-switch-blocks",
            "methods": {
                "cpu_backup": _block_provenance(cpu_blocks, kind="same-process-vllm-switch-cpu"),
                "exact_disk": _block_provenance(
                    disk_blocks, kind="same-process-vllm-switch-exact-disk"
                ),
            },
            "benchmark_repo": _git_identity(cpu_blocks[0]["environment"]["benchmark_repo"]),
            "engine_repo": _git_identity(cpu_blocks[0]["environment"]["vllm_repo"]),
            "engine_runtime": cpu_blocks[0]["environment"]["runtime"],
            "model_identity": {"identity": cpu_blocks[0]["model"]},
        },
    }
    document = {
        "schema_version": 3,
        "title": "Qwen2.5-0.5B sleep and wake latency profiling",
        "metric_boundary": METRIC_BOUNDARY,
        "model": "Qwen2.5-0.5B-Instruct",
        "frozen_scope": {
            "gpu": "NVIDIA GeForce RTX 3080 10 GiB",
            "dtype": "float16",
            "max_model_len": 1024,
            "gpu_memory_utilization": 0.8,
            "engine_mode": "eager",
            "process_blocks_per_method": 3,
            "cycles_per_process": 3,
            "sample_count_per_method": 3,
        },
        "stability_rule": {
            "process_block": "three independent engine processes per mechanism",
            "first": "cycle index zero in each process block",
            "steady": "arithmetic mean of the two steady cycles within each process block",
            "l2_cache": (
                "one validated cold cycle and the arithmetic mean of two validated warm cycles "
                "within each block"
            ),
            "center": "median of three process-block observations",
            "spread": "minimum and maximum of three process-block observations",
            "profile": "real process-block sample nearest the operation median",
        },
        "phase_semantics": PHASE_SEMANTICS,
        "sources": list(sources.values()),
        "source_provenance": source_provenance,
        "samples": samples,
        "evidence_label": "controlled local mechanism comparison",
        "comparability": {
            "shared_conditions": [
                "Qwen2.5-0.5B-Instruct",
                "NVIDIA RTX 3080",
                "float16",
                "max_model_len=1024",
                "gpu_memory_utilization=0.80",
                "eager execution",
                "three independent process blocks with three cycles each",
                "continuous wake envelope and identical phase accounting",
            ],
            "cache_conditions": {
                "vLLM L2 Cold": (
                    "per-file POSIX_FADV_DONTNEED before untimed mincore validation; "
                    "near-checkpoint physical reads required during timed wake"
                ),
                "vLLM L2 Warm": (
                    "no eviction; high mincore residency and negligible physical reads required"
                ),
            },
            "prohibited_claim": (
                "Do not treat cycles within one process as independent observations."
            ),
        },
    }
    aggregate_profiles(document)
    return document
