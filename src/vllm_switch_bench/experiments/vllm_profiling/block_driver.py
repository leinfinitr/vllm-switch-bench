"""Execute repeated in-process vLLM sleep/wake profiling blocks."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vllm_switch_bench.common.provenance import git_metadata
from vllm_switch_bench.common.resources import (
    process_tree_rss_bytes,
    query_gpu_memory_used_mib,
    read_meminfo_bytes,
)
from vllm_switch_bench.common.schema import PROMPTS
from vllm_switch_bench.experiments.vllm_profiling.page_cache import (
    checkpoint_files,
    evict_page_cache,
    l2_cache_schedule,
    measure_page_cache,
    process_tree_io_delta,
    process_tree_io_snapshot,
    validate_cache_observation,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default="bench-model")
    parser.add_argument(
        "--method",
        choices=["sleep_l1", "sleep_l2", "cpu_backup", "exact_disk"],
        required=True,
    )
    parser.add_argument("--process-block", type=int, required=True)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="short_short")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--extra-vllm-arg", action="append", default=[])
    parser.add_argument("--idle-s", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cold-max-resident-ratio", type=float, default=0.05)
    parser.add_argument("--cold-min-read-ratio", type=float, default=0.90)
    parser.add_argument("--warm-min-resident-ratio", type=float, default=0.90)
    parser.add_argument("--warm-max-read-ratio", type=float, default=0.10)
    args = parser.parse_args(argv)
    if args.process_block < 0:
        parser.error("--process-block must be non-negative")
    if args.cycles < 2:
        parser.error("--cycles must be at least two")
    return args


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _events(path: Path, start: float, end: float) -> list[dict[str, Any]]:
    return [
        event
        for event in _read_jsonl(path)
        if start <= float(event.get("monotonic_s", -1.0)) <= end
    ]


def _phase(events: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    matching = [event for event in events if event.get("phase") == name]
    return matching[-1] if matching else None


def _runtime_metadata() -> dict[str, Any]:
    import torch
    import vllm

    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm_version": getattr(vllm, "__version__", None),
        "vllm_import_path": str(Path(str(vllm.__file__)).resolve()),
    }


def _infer(llm, sampling_params, prompt: str) -> dict[str, Any]:
    started = time.perf_counter()
    outputs = llm.generate(prompt, sampling_params, use_tqdm=False)
    latency = time.perf_counter() - started
    output = outputs[0].outputs[0]
    return {
        "latency_s": latency,
        "token_ids": [int(token) for token in output.token_ids],
        "text": output.text,
    }


def _sleep_profile(path: Path, start: float, end: float) -> dict[str, Any]:
    events = _events(path, start, end)
    return {
        "started_monotonic_s": start,
        "ended_monotonic_s": end,
        "events": events,
    }


def _l2_restore(llm, profile_path: Path) -> dict[str, Any]:
    total_start = time.perf_counter()

    wake_weights_start = time.perf_counter()
    llm.wake_up(tags=["weights"])
    wake_weights_end = time.perf_counter()

    reload_start = time.perf_counter()
    llm.collective_rpc("reload_weights")
    reload_end = time.perf_counter()

    wake_kv_start = time.perf_counter()
    llm.wake_up(tags=["kv_cache"])
    wake_kv_end = time.perf_counter()

    total_end = time.perf_counter()
    steps = {
        "wake_weights": {
            "latency_s": wake_weights_end - wake_weights_start,
            "sleep_profile": _sleep_profile(profile_path, wake_weights_start, wake_weights_end),
        },
        "reload_weights": {
            "latency_s": reload_end - reload_start,
            "sleep_profile": _sleep_profile(profile_path, reload_start, reload_end),
        },
        "wake_kv_cache": {
            "latency_s": wake_kv_end - wake_kv_start,
            "sleep_profile": _sleep_profile(profile_path, wake_kv_start, wake_kv_end),
        },
    }
    active = sum(step["latency_s"] for step in steps.values())
    return {
        "latency_s": total_end - total_start,
        "active_latency_s": active,
        "inter_step_gap_s": max(0.0, total_end - total_start - active),
        "steps": steps,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    gpu_used_before_mib = query_gpu_memory_used_mib()
    if gpu_used_before_mib is not None and gpu_used_before_mib > 32:
        raise RuntimeError(
            f"GPU is not idle before process block: {gpu_used_before_mib} MiB in use"
        )
    profile_path = args.out_dir / "sleep_profile.jsonl"
    args.out_dir.mkdir(parents=True, exist_ok=False)
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_SLEEP_PROFILE_PATH"] = str(profile_path)
    if args.method == "exact_disk":
        backup_root = args.out_dir / "exact-disk-backup"
        os.environ["VLLM_EXACT_DISK_BACKUP_ENABLED"] = "1"
        os.environ["VLLM_EXACT_DISK_BACKUP_DIR"] = str(backup_root)
        os.environ["VLLM_EXACT_DISK_BACKUP_DIRECT_IO"] = "1"

    from vllm import LLM, SamplingParams  # type: ignore[import-not-found]

    prompt = str(PROMPTS[args.prompt]["prompt"])
    sampling_params = SamplingParams(
        max_tokens=int(PROMPTS[args.prompt]["max_tokens"]), temperature=0.0, seed=0
    )
    load_started = time.perf_counter()
    extra_kwargs: dict[str, Any] = {}
    for value in args.extra_vllm_arg:
        name, separator, raw = value.partition("=")
        if not separator:
            raise ValueError("block-driver --extra-vllm-arg values must use KEY=VALUE syntax")
        normalized = name.strip().replace("-", "_")
        text = raw.strip()
        if text.lower() in {"true", "false"}:
            parsed: Any = text.lower() == "true"
        else:
            try:
                parsed = int(text)
            except ValueError:
                try:
                    parsed = float(text)
                except ValueError:
                    parsed = text
        extra_kwargs[normalized] = parsed
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        enable_sleep_mode=True,
        enforce_eager=args.enforce_eager,
        **extra_kwargs,
    )
    load_latency = time.perf_counter() - load_started
    before = _infer(llm, sampling_params, prompt)
    expected = (before["token_ids"], before["text"])
    demotion = None
    demotion_latency_s: float | None = None
    if args.method == "exact_disk":
        demotion_started = time.perf_counter()
        demotion = llm.collective_rpc("demote_weight_cpu_backup_to_disk")
        demotion_latency_s = time.perf_counter() - demotion_started
        if not demotion or not all(
            int(result.get("remaining_cpu_backup_bytes", -1)) == 0
            and int(result.get("pending_release_bytes", -1)) == 0
            and int(result.get("released_bytes_total", 0)) > 0
            for result in demotion
        ):
            raise RuntimeError(f"exact-disk demotion did not complete: {demotion!r}")
    files = checkpoint_files(args.model) if args.method == "sleep_l2" else []
    conditions = (
        l2_cache_schedule(args.process_block, args.cycles)
        if args.method == "sleep_l2"
        else [None] * args.cycles
    )
    cycles: list[dict[str, Any]] = []
    for cycle_index, condition in enumerate(conditions):
        sleep_started = time.perf_counter()
        llm.sleep(level=2 if args.method == "sleep_l2" else 1)
        sleep_ended = time.perf_counter()
        time.sleep(args.idle_s)

        cache: dict[str, Any] | None = None
        if condition == "cold":
            eviction = evict_page_cache(files)
            cache = {
                "condition": condition,
                "treatment": "posix_fadvise_dontneed",
                "eviction": eviction,
                "before_wake": eviction["after"],
            }
        elif condition == "warm":
            cache = {
                "condition": condition,
                "treatment": "none",
                "before_wake": measure_page_cache(files),
            }

        pid = os.getpid()
        io_before = process_tree_io_snapshot(pid) if cache else None
        if args.method == "sleep_l2":
            restore = _l2_restore(llm, profile_path)
        else:
            wake_started = time.perf_counter()
            llm.wake_up()
            wake_ended = time.perf_counter()
            restore = {
                "latency_s": wake_ended - wake_started,
                "sleep_profile": _sleep_profile(profile_path, wake_started, wake_ended),
            }
        io_after = process_tree_io_snapshot(pid) if cache else None
        cache_valid = True
        cache_failures: list[str] = []
        if cache is not None:
            io_delta = process_tree_io_delta(io_before or {}, io_after or {})
            checkpoint_bytes = int(cache["before_wake"]["total_bytes"])
            cache_valid, cache_failures = validate_cache_observation(
                str(condition),
                before_wake=cache["before_wake"],
                io_delta=io_delta,
                checkpoint_bytes=checkpoint_bytes,
                cold_max_resident_ratio=args.cold_max_resident_ratio,
                cold_min_read_ratio=args.cold_min_read_ratio,
                warm_min_resident_ratio=args.warm_min_resident_ratio,
                warm_max_read_ratio=args.warm_max_read_ratio,
            )
            cache.update(
                {
                    "io_before": io_before,
                    "io_after": io_after,
                    "io_delta": io_delta,
                    "storage_read_ratio": (
                        io_delta["read_bytes"] / checkpoint_bytes if checkpoint_bytes else 0.0
                    ),
                    "valid": cache_valid,
                    "failures": cache_failures,
                }
            )
        after = _infer(llm, sampling_params, prompt)
        output_equal = expected == (after["token_ids"], after["text"])
        sleep_profile = _sleep_profile(profile_path, sleep_started, sleep_ended)
        cycles.append(
            {
                "cycle_index": cycle_index,
                "cycle_class": "first" if cycle_index == 0 else "steady",
                "cache_condition": condition,
                "sleep": {
                    "latency_s": sleep_ended - sleep_started,
                    "sleep_profile": sleep_profile,
                },
                "restore": restore,
                "cache_observation": cache,
                "infer_after": after,
                "output_equal": output_equal,
                "ok": output_equal and cache_valid,
            }
        )

    benchmark_root_value = os.environ.get("VLLM_SWITCH_BENCH_ROOT")
    if not benchmark_root_value:
        raise RuntimeError("VLLM_SWITCH_BENCH_ROOT is required for block provenance")
    benchmark_root = Path(benchmark_root_value).resolve(strict=True)
    return {
        "schema_version": 1,
        "experiment": "vllm-profiling-block",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": args.method,
        "process_block": args.process_block,
        "cycles_per_process": args.cycles,
        "model": str(Path(args.model).resolve()),
        "startup_latency_s": load_latency,
        "gpu_used_before_mib": gpu_used_before_mib,
        "infer_before": before,
        "exact_disk_demotion": demotion,
        "exact_disk_demotion_latency_s": demotion_latency_s,
        "cycles": cycles,
        "ok": all(cycle["ok"] for cycle in cycles),
        "environment": {
            "benchmark_repo": git_metadata(benchmark_root),
            "vllm_repo": git_metadata(Path.cwd()),
            "runtime": _runtime_metadata(),
            "platform": platform.platform(),
            "initial_meminfo_bytes": read_meminfo_bytes(),
            "process_tree_rss_bytes": process_tree_rss_bytes(os.getpid()),
        },
        "parameters": {
            "prompt": args.prompt,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "dtype": args.dtype,
            "enforce_eager": args.enforce_eager,
            "idle_s": args.idle_s,
            "cold_max_resident_ratio": args.cold_max_resident_ratio,
            "cold_min_read_ratio": args.cold_min_read_ratio,
            "warm_min_resident_ratio": args.warm_min_resident_ratio,
            "warm_max_read_ratio": args.warm_max_read_ratio,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result: dict[str, Any]
    try:
        result = run(args)
    except BaseException as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "schema_version": 1,
            "experiment": "vllm-profiling-block",
            "method": args.method,
            "process_block": args.process_block,
            "ok": False,
            "error": repr(exc),
        }
    output = args.out_dir / "block-summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
