#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from benchlib.schema import PROMPTS


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: str


def parse_model_spec(value: str) -> ModelSpec:
    if "=" not in value:
        path = value
        name = Path(path).name.replace(".", "_").replace("-", "_")
        return ModelSpec(name=name, path=path)
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError(
            "model spec must be NAME=/path/to/model or /path/to/model"
        )
    return ModelSpec(name=name, path=path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repeatedly load, infer, and sleep two vLLM models to measure "
            "CPU backup pool behavior across model lifecycles."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=parse_model_spec,
        default=[
            ModelSpec("qwen2p5_0p5b", "/home/ljl/models/hf/Qwen2.5-0.5B-Instruct"),
            ModelSpec("qwen2p5_1p5b", "/home/ljl/models/hf/Qwen2.5-1.5B-Instruct"),
        ],
        help="Models in NAME=PATH form. The sequence is repeated in this order.",
    )
    parser.add_argument("--out-dir", default="results/profiling/phase1_pinned_pool")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="short_short")
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of times to repeat the full model sequence.",
    )
    parser.add_argument(
        "--coordinator-url",
        default=None,
        help=(
            "Optional CPU backup metadata coordinator URL, e.g. "
            "http://127.0.0.1:9000. When set, vLLM reports local backup "
            "metadata to the daemon while keeping memory/copy local."
        ),
    )
    parser.add_argument(
        "--coordinator-timeout-s",
        type=float,
        default=0.05,
        help="Per-flush coordinator HTTP timeout inside vLLM workers.",
    )
    parser.add_argument(
        "--post-wake-observation-s",
        type=float,
        default=0.0,
        help=(
            "Seconds to observe host/process memory after wake-up. This makes "
            "physical pinned-memory reclaim visible without changing the "
            "default benchmark behavior."
        ),
    )
    return parser.parse_args(argv)


def load_profile_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def events_in_window(
    events: list[dict[str, Any]],
    phase: str,
    start_monotonic_s: float,
    end_monotonic_s: float,
) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if event.get("phase") == phase
        and start_monotonic_s <= float(event.get("monotonic_s", -1.0)) <= end_monotonic_s
    ]


def newest_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not events:
        return None
    return max(events, key=lambda event: float(event.get("monotonic_s", 0.0)))


def read_meminfo_bytes() -> dict[str, int]:
    """Read host memory counters used to validate physical reclaim."""
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        name, raw = line.split(":", 1)
        fields = raw.split()
        if fields:
            values[name] = int(fields[0]) * 1024
    return values


def read_process_memory_bytes(pid: int) -> dict[str, int]:
    """Read lightweight process RSS counters without extra dependencies."""
    values: dict[str, int] = {}
    status_path = Path(f"/proc/{pid}/status")
    if not status_path.exists():
        return values
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        if name not in {"VmRSS", "RssAnon", "RssFile", "VmLck"}:
            continue
        fields = raw.split()
        if fields:
            values[name] = int(fields[0]) * 1024
    return values


def record_memory_snapshot(step: dict[str, Any], prefix: str, pid: int | None) -> None:
    host = read_meminfo_bytes()
    for name in ("MemTotal", "MemAvailable", "MemFree", "Unevictable", "Mlocked"):
        if name in host:
            step[f"{prefix}_host_{name.lower()}_bytes"] = host[name]
    if pid is None:
        return
    step[f"{prefix}_worker_pid"] = pid
    for name, value in read_process_memory_bytes(pid).items():
        step[f"{prefix}_worker_{name.lower()}_bytes"] = value


def flatten_breakdown(prefix: str, event: dict[str, Any] | None) -> dict[str, Any]:
    if event is None:
        return {}
    fields = [
        "phase",
        "latency_s",
        "copy_d2h_s",
        "copy_h2d_s",
        "create_map_s",
        "unmap_release_s",
        "cpu_backup_alloc_s",
        "cpu_backup_pool_hit_count",
        "cpu_backup_pool_miss_count",
        "cpu_backup_reuse_count",
        "cpu_backup_reused_bytes",
        "cpu_backup_pool_reserved_bytes",
        "cpu_backup_pool_free_bytes",
        "allocation_count",
        "total_bytes",
        "backup_bytes",
        "discard_bytes",
        "bytes",
        "cpu_backup_coordinator_enabled",
        "cpu_backup_coordinator_backend",
        "cpu_backup_coordinator_events_sent",
        "cpu_backup_coordinator_flush_errors",
        "cpu_backup_coordinator_pending_events",
        "cpu_backup_coordinator_eviction_polls",
        "cpu_backup_coordinator_eviction_requests_received",
        "cpu_backup_eviction_released_count",
        "cpu_backup_eviction_released_bytes",
        "cpu_backup_host_cache_flush_count",
        "cpu_backup_host_cache_flush_errors",
    ]
    row: dict[str, Any] = {}
    for field in fields:
        if field in event:
            row[f"{prefix}_{field}"] = event[field]
    for field in [
        "bytes_by_tag",
        "backup_bytes_by_tag",
        "discard_bytes_by_tag",
        "restored_bytes_by_tag",
        "remapped_without_backup_bytes_by_tag",
    ]:
        if field in event:
            row[f"{prefix}_{field}"] = json.dumps(event[field], sort_keys=True)
    return row


def write_steps_csv(path: Path, steps: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for step in steps:
        for key in step:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(steps)


def model_load_kwargs(args: argparse.Namespace, model: ModelSpec) -> dict[str, Any]:
    return {
        "model": model.path,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "dtype": args.dtype,
        "enable_sleep_mode": True,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["VLLM_USE_V1"] = "1"

    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"phase1_two_model_repeated_sleep_l1_{int(time.time())}"
    profile_path = out_dir / f"{run_id}.sleep_profile.jsonl"
    os.environ["VLLM_SLEEP_PROFILE_PATH"] = str(profile_path.resolve())
    if args.coordinator_url:
        os.environ["VLLM_CPU_BACKUP_COORDINATOR"] = "daemon"
        os.environ["VLLM_CPU_BACKUP_COORDINATOR_URL"] = args.coordinator_url
        os.environ["VLLM_CPU_BACKUP_COORDINATOR_TIMEOUT_S"] = str(
            args.coordinator_timeout_s
        )
    else:
        os.environ.pop("VLLM_CPU_BACKUP_COORDINATOR", None)
        os.environ.pop("VLLM_CPU_BACKUP_COORDINATOR_URL", None)
        os.environ.pop("VLLM_CPU_BACKUP_COORDINATOR_TIMEOUT_S", None)

    summary: dict[str, Any] = {
        "models": [{"name": model.name, "path": model.path} for model in args.models],
        "iterations": args.iterations,
        "profile_path": str(profile_path),
        "out_dir": str(out_dir),
        "coordinator_url": args.coordinator_url,
        "post_wake_observation_s": args.post_wake_observation_s,
    }
    steps: list[dict[str, Any]] = []

    try:
        from vllm import LLM, SamplingParams

        prompt_spec = PROMPTS[args.prompt]
        prompt = prompt_spec["prompt"]
        sampling_params = SamplingParams(
            max_tokens=prompt_spec["max_tokens"], temperature=0.0, seed=0
        )

        engines: dict[str, LLM] = {}
        engine_pids: dict[str, int] = {}
        try:
            for iteration in range(args.iterations):
                for model_index, model in enumerate(args.models):
                    step_index = iteration * len(args.models) + model_index
                    step: dict[str, Any] = {
                        "step_index": step_index,
                        "iteration": iteration,
                        "model_index": model_index,
                        "model_name": model.name,
                        "model_path": model.path,
                    }
                    try:
                        if model.name not in engines:
                            if args.coordinator_url:
                                os.environ["VLLM_CPU_BACKUP_COORDINATOR_MODEL_ID"] = model.name
                                os.environ["VLLM_CPU_BACKUP_COORDINATOR_CLIENT_ID"] = (
                                    f"{run_id}:{model.name}"
                                )
                            activate_started = time.perf_counter()
                            engines[model.name] = LLM(**model_load_kwargs(args, model))
                            activate_latency_s = time.perf_counter() - activate_started
                            step["activate_type"] = "load"
                            step["load_latency_s"] = activate_latency_s
                            step["activate_latency_s"] = activate_latency_s
                        else:
                            llm = engines[model.name]
                            worker_pid = engine_pids.get(model.name)
                            record_memory_snapshot(step, "pre_wake", worker_pid)
                            activate_started = time.perf_counter()
                            llm.wake_up()
                            activate_latency_s = time.perf_counter() - activate_started
                            step["activate_type"] = "wake"
                            step["wake_latency_s"] = activate_latency_s
                            step["activate_latency_s"] = activate_latency_s
                            if args.post_wake_observation_s > 0:
                                time.sleep(args.post_wake_observation_s)
                            record_memory_snapshot(step, "post_wake", worker_pid)

                        llm = engines[model.name]
                        infer_started = time.perf_counter()
                        outputs = llm.generate(prompt, sampling_params, use_tqdm=False)
                        step["infer_latency_s"] = time.perf_counter() - infer_started
                        step["output_text"] = outputs[0].outputs[0].text

                        events_before_sleep = load_profile_events(profile_path)
                        sleep_started = time.perf_counter()
                        llm.sleep(level=1)
                        sleep_ended = time.perf_counter()
                        step["sleep_latency_s"] = sleep_ended - sleep_started
                        events_after_sleep = load_profile_events(profile_path)
                        new_events = events_after_sleep[len(events_before_sleep) :]
                        allocator_sleep = newest_event(
                            events_in_window(
                                new_events,
                                "allocator_sleep",
                                sleep_started,
                                sleep_ended,
                            )
                        )
                        step.update(flatten_breakdown("sleep_allocator", allocator_sleep))
                        if allocator_sleep is not None and "pid" in allocator_sleep:
                            engine_pids[model.name] = int(allocator_sleep["pid"])
                        step["ok"] = True
                    except Exception as exc:
                        step["ok"] = False
                        step["error"] = repr(exc)
                        steps.append(step)
                        raise
                    steps.append(step)
        finally:
            engines.clear()
            gc.collect()

        summary["ok"] = all(step.get("ok") for step in steps)
        summary["steps"] = steps
        summary["sleep_profile_events"] = load_profile_events(profile_path)
    except Exception as exc:
        summary["ok"] = False
        summary["error"] = repr(exc)
        summary["steps"] = steps
    finally:
        summary_path = out_dir / "phase1_two_model_repeated_sleep_summary.json"
        steps_csv_path = out_dir / "phase1_two_model_repeated_sleep_steps.csv"
        summary["summary_path"] = str(summary_path)
        summary["steps_csv_path"] = str(steps_csv_path)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        write_steps_csv(steps_csv_path, steps)
        print(summary_path)
    return 0 if summary.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
