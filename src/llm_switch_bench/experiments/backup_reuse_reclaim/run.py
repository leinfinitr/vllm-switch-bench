#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_switch_bench.common.environment import reexec_with_python
from llm_switch_bench.common.provenance import file_metadata, repository_root
from llm_switch_bench.common.resources import (
    process_tree_rss_bytes,
    read_meminfo_bytes,
    read_process_memory_bytes,
)
from llm_switch_bench.common.schema import PROMPTS


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: str


INVOCATION_DIR_ENV = "LLM_SWITCH_BENCH_INVOCATION_DIR"


def invocation_path(value: str | Path, invocation_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else invocation_dir / path


def normalize_invocation_paths(args: argparse.Namespace) -> None:
    """Keep CLI paths anchored when execution moves into the vLLM checkout."""

    invocation_dir = Path(os.environ.setdefault(INVOCATION_DIR_ENV, str(Path.cwd().resolve())))
    args.python = str(invocation_path(args.python, invocation_dir).absolute())
    args.vllm_repo = invocation_path(args.vllm_repo, invocation_dir).absolute()
    args.out_dir = str(invocation_path(args.out_dir, invocation_dir).absolute())
    args.models = [
        ModelSpec(model.name, str(invocation_path(model.path, invocation_dir).absolute()))
        for model in args.models
    ]
    if args.coordinator_repo:
        args.coordinator_repo = invocation_path(args.coordinator_repo, invocation_dir).absolute()
    if args.coordinator_config:
        args.coordinator_config = invocation_path(
            args.coordinator_config, invocation_dir
        ).absolute()


def parse_model_spec(value: str) -> ModelSpec:
    if "=" not in value:
        path = value
        name = Path(path).name.replace(".", "_").replace("-", "_")
        return ModelSpec(name=name, path=path)
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("model spec must be NAME=/path/to/model or /path/to/model")
    return ModelSpec(name=name, path=path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repeatedly load, infer, and sleep vLLM models to measure "
            "CPU backup pool behavior across model lifecycles."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=parse_model_spec,
        required=True,
        help="Models in NAME=PATH form. The sequence is repeated in this order.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable from the vLLM environment used for this measurement.",
    )
    parser.add_argument(
        "--vllm-repo",
        type=Path,
        required=True,
        help="Checkout that must provide the imported vLLM package.",
    )
    parser.add_argument("--out-dir", default="results/tmp/backup-reuse-reclaim")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable compilation and CUDA graphs for a comparable eager-mode run.",
    )
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
            "Optional aggregate CPU backup coordinator URL, e.g. "
            "http://127.0.0.1:9000. vLLM keeps tensor ownership local."
        ),
    )
    parser.add_argument(
        "--coordinator-repo",
        type=Path,
        help="Checkout that provides the CPU backup coordinator service.",
    )
    parser.add_argument(
        "--coordinator-config",
        type=Path,
        help="Exact coordinator configuration used for this measurement.",
    )
    parser.add_argument(
        "--coordinator-timeout-s",
        type=float,
        default=0.05,
        help="Per-flush coordinator HTTP timeout inside vLLM workers.",
    )
    parser.add_argument(
        "--post-release-observation-s",
        "--post-wake-observation-s",
        dest="post_release_observation_s",
        type=float,
        default=0.0,
        help=(
            "Seconds to observe host/process memory after a lifecycle operation. "
            "The deprecated --post-wake-observation-s spelling is retained as an alias."
        ),
    )
    parser.add_argument(
        "--expect-release",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Assert whether coordinator-driven backup release should occur.",
    )
    parser.add_argument(
        "--expect-reuse",
        action="store_true",
        help="Require a later sleep to reuse backup bytes with zero D2H time.",
    )
    parser.add_argument(
        "--min-worker-rss-reclaim-bytes",
        type=int,
        default=0,
        help="Minimum worker RSS decrease required on a released wake step.",
    )
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if (args.expect_release is not None or args.expect_reuse) and args.iterations < 2:
        parser.error("release/reuse assertions require --iterations at least 2")
    if args.post_release_observation_s < 0:
        parser.error("--post-release-observation-s must be non-negative")
    if args.min_worker_rss_reclaim_bytes < 0:
        parser.error("--min-worker-rss-reclaim-bytes must be non-negative")
    if args.expect_release is not True and args.min_worker_rss_reclaim_bytes > 0:
        parser.error("--min-worker-rss-reclaim-bytes requires --expect-release")
    if args.expect_release is True and args.expect_reuse:
        parser.error("--expect-release and --expect-reuse are mutually exclusive")
    coordinator_identity = (args.coordinator_repo, args.coordinator_config)
    if args.coordinator_url and not all(coordinator_identity):
        parser.error("--coordinator-url requires --coordinator-repo and --coordinator-config")
    if not args.coordinator_url and any(coordinator_identity):
        parser.error("--coordinator-repo and --coordinator-config require --coordinator-url")
    model_names = [model.name for model in args.models]
    if len(model_names) != len(set(model_names)):
        parser.error("--models names must be unique")
    return args


def load_profile_events_since(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    """Read only newly appended JSONL records and return the next byte offset."""
    if not path.exists():
        return [], offset
    size = path.stat().st_size
    if size < offset:
        # A restarted writer truncated/replaced the profile file.
        offset = 0
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        while True:
            line_offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith("\n"):
                # Do not consume a record while the writer is still appending it.
                handle.seek(line_offset)
                break
            if line.strip():
                events.append(json.loads(line))
        return events, handle.tell()


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


def record_memory_snapshot(step: dict[str, Any], prefix: str, pid: int | None) -> None:
    host = read_meminfo_bytes()
    for name in ("MemTotal", "MemAvailable", "MemFree", "Unevictable", "Mlocked"):
        if name in host:
            step[f"{prefix}_host_{name.lower()}_bytes"] = host[name]
    tree_rss = process_tree_rss_bytes(os.getpid())
    if tree_rss is not None:
        step[f"{prefix}_process_tree_rss_bytes"] = tree_rss
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
        "cpu_backup_coordinator_requests_succeeded",
        "cpu_backup_coordinator_request_errors",
        "cpu_backup_coordinator_pending_usage",
        "cpu_backup_coordinator_release_polls",
        "cpu_backup_coordinator_release_bytes_received",
        "cpu_backup_release_count",
        "cpu_backup_release_bytes",
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


def record_release_delta(
    step: dict[str, Any],
    prefix: str,
    event: dict[str, Any] | None,
    model_name: str,
    last_release_bytes: dict[str, int],
) -> None:
    if event is None:
        return
    cumulative = int(event.get("cpu_backup_release_bytes", 0) or 0)
    delta = max(cumulative - last_release_bytes.get(model_name, 0), 0)
    step[f"{prefix}_cpu_backup_release_delta_bytes"] = delta
    step["cpu_backup_release_delta_bytes"] = (
        int(step.get("cpu_backup_release_delta_bytes", 0) or 0) + delta
    )
    last_release_bytes[model_name] = cumulative


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


def record_evidence_errors(summary: dict[str, Any], evidence_errors: list[str]) -> None:
    """Fail closed when a run cannot preserve its required audit evidence."""
    if not evidence_errors:
        return
    summary["evidence_collection_errors"] = evidence_errors
    summary["ok"] = False


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def prepend_python_bin_to_path(python_executable: str | Path = sys.executable) -> Path:
    """Expose tools installed beside the active interpreter to worker children."""

    # Do not resolve a virtual-environment Python symlink: its parent is the
    # environment's bin directory, which contains JIT helpers such as ninja.
    python_bin = Path(python_executable).expanduser().absolute().parent
    entries = os.environ.get("PATH", "").split(os.pathsep)
    os.environ["PATH"] = os.pathsep.join(
        [str(python_bin), *(entry for entry in entries if entry != str(python_bin))]
    )
    return python_bin


def fetch_run_coordinator_stats(base_url: str, run_id: str, timeout_s: float) -> dict[str, Any]:
    """Return protocol accounting for this benchmark run's worker clients."""
    request = urllib.request.Request(f"{base_url.rstrip('/')}/admin/cpu-backup/stats", method="GET")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    clients = payload.get("clients")
    if not isinstance(clients, dict):
        raise ValueError("coordinator stats response is missing clients")
    run_clients = {
        client_id: record
        for client_id, record in clients.items()
        if client_id.startswith(f"{run_id}:") and isinstance(record, dict)
    }
    return {
        "client_count": len(run_clients),
        "requested_release_bytes_total": sum(
            int(record.get("requested_release_bytes_total", 0) or 0)
            for record in run_clients.values()
        ),
        "released_bytes_total": sum(
            int(record.get("released_bytes_total", 0) or 0) for record in run_clients.values()
        ),
        "pending_release_bytes": sum(
            int(record.get("pending_release_bytes", 0) or 0) for record in run_clients.values()
        ),
        "clients": run_clients,
    }


def wait_for_run_coordinator_stats(
    base_url: str,
    run_id: str,
    *,
    timeout_s: float,
    expect_release: bool,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while True:
        stats = fetch_run_coordinator_stats(base_url, run_id, min(timeout_s, 1.0))
        settled = stats["pending_release_bytes"] == 0 and (
            stats["released_bytes_total"] >= stats["requested_release_bytes_total"]
        )
        released = stats["released_bytes_total"] > 0
        if settled and (not expect_release or released):
            return stats
        if time.monotonic() >= deadline:
            return stats
        time.sleep(0.1)


def record_coordinator_snapshot(
    step: dict[str, Any],
    prefix: str,
    base_url: str,
    run_id: str,
    timeout_s: float,
) -> None:
    """Record run-local counters at a physical observation boundary."""
    stats = fetch_run_coordinator_stats(base_url, run_id, timeout_s)
    for key in (
        "requested_release_bytes_total",
        "released_bytes_total",
        "pending_release_bytes",
    ):
        step[f"{prefix}_coordinator_{key}"] = int(stats[key])


def wake_reclaim_delta(step: dict[str, Any]) -> int:
    """Return bytes the coordinator observed as released across one wake window."""
    before = step.get("pre_wake_coordinator_released_bytes_total")
    after = step.get("post_wake_coordinator_released_bytes_total")
    if before is None or after is None:
        return 0
    return int(after) - int(before)


def git_metadata(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(path), *args],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    return {
        "repo_path": str(path.resolve()),
        "git_commit": run("rev-parse", "HEAD"),
        "git_branch": run("branch", "--show-current"),
        "git_dirty": bool(run("status", "--porcelain")),
        # Raw benchmark output is often intentionally untracked inside the repo.
        # Distinguish that from source/index modifications for provenance.
        "git_tracked_dirty": bool(
            run("diff", "--name-only") or run("diff", "--cached", "--name-only")
        ),
    }


def module_repo_metadata(module_name: str) -> dict[str, Any] | None:
    """Return the imported module path and enclosing Git checkout, when available."""
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return None
    module_path = Path(spec.origin).resolve()
    for candidate in (module_path.parent, *module_path.parents):
        if (candidate / ".git").exists():
            metadata = git_metadata(candidate)
            metadata["module_path"] = str(module_path)
            return metadata
    return {"module_path": str(module_path), "git_commit": None}


def validate_results(
    args: argparse.Namespace,
    steps: list[dict[str, Any]],
    coordinator_stats: dict[str, Any] | None = None,
) -> list[str]:
    """Return assertion failures for release/no-release experimental controls."""
    failures: list[str] = []
    release_steps = [
        step for step in steps if int(step.get("cpu_backup_release_delta_bytes", 0) or 0) > 0
    ]
    if args.expect_release is True and not release_steps:
        failures.append("expected coordinator-driven release but observed none")
    if args.expect_release is False and release_steps:
        failures.append("expected no release but observed released bytes")
    mismatched_outputs = [step for step in steps if step.get("output_matches_reference") is False]
    if mismatched_outputs:
        failures.append("deterministic model output changed after sleep/wake")
    if args.expect_release is True:
        flush_error_fields = [
            int(value or 0)
            for step in release_steps
            for key, value in step.items()
            if key.endswith("cpu_backup_host_cache_flush_errors")
        ]
        if not flush_error_fields:
            failures.append("release steps are missing host-cache flush telemetry")
        elif any(value != 0 for value in flush_error_fields):
            failures.append("host-cache flush reported one or more errors")
        if coordinator_stats is None:
            failures.append("release assertion is missing final coordinator stats")
        else:
            requested = int(coordinator_stats.get("requested_release_bytes_total", 0) or 0)
            released = int(coordinator_stats.get("released_bytes_total", 0) or 0)
            pending = int(coordinator_stats.get("pending_release_bytes", 0) or 0)
            if requested <= 0:
                failures.append("coordinator reported no release request for this run")
            # The allocator releases whole tensors/blocks, so a safe candidate
            # may legitimately overshoot a byte target. Under-release or any
            # outstanding obligation is still a failure.
            if released < requested or pending != 0:
                failures.append(
                    "release protocol did not settle: "
                    f"requested={requested}, released={released}, pending={pending}"
                )
    elif args.expect_release is False and coordinator_stats is not None:
        client_count = int(coordinator_stats.get("client_count", 0) or 0)
        requested = int(coordinator_stats.get("requested_release_bytes_total", 0) or 0)
        released = int(coordinator_stats.get("released_bytes_total", 0) or 0)
        pending = int(coordinator_stats.get("pending_release_bytes", 0) or 0)
        if client_count <= 0:
            failures.append("no-pressure control has no run-local coordinator client")
        if requested != 0 or released != 0 or pending != 0:
            failures.append(
                "no-pressure control observed coordinator release activity: "
                f"requested={requested}, released={released}, pending={pending}"
            )
    if args.expect_reuse:
        reuse_steps = [
            step
            for step in steps
            if int(step.get("sleep_allocator_cpu_backup_reused_bytes", 0) or 0) > 0
            and float(step.get("sleep_allocator_copy_d2h_s", -1.0)) == 0.0
        ]
        if not reuse_steps:
            failures.append("expected backup reuse with zero D2H time but observed none")
    if args.min_worker_rss_reclaim_bytes > 0:
        reclaim_steps = [step for step in steps if wake_reclaim_delta(step) > 0]
        if not reclaim_steps:
            failures.append(
                "physical reclaim assertion requires a wake window with a positive "
                "coordinator release delta"
            )
        for step in reclaim_steps:
            before = step.get("pre_wake_process_tree_rss_bytes")
            after = step.get("post_wake_process_tree_rss_bytes")
            mem_before = step.get("pre_wake_host_memavailable_bytes")
            mem_after = step.get("post_wake_host_memavailable_bytes")
            if before is None or after is None or mem_before is None or mem_after is None:
                failures.append("release settlement window is missing physical-memory snapshots")
                continue
            reclaimed = int(before) - int(after)
            if reclaimed < args.min_worker_rss_reclaim_bytes:
                failures.append(
                    f"worker RSS reclaimed {reclaimed} bytes, below required "
                    f"{args.min_worker_rss_reclaim_bytes}"
                )
            available_delta = int(mem_after) - int(mem_before)
            if available_delta < args.min_worker_rss_reclaim_bytes:
                failures.append(
                    f"host MemAvailable increased {available_delta} bytes, below required "
                    f"{args.min_worker_rss_reclaim_bytes}"
                )
    return failures


def model_load_kwargs(args: argparse.Namespace, model: ModelSpec) -> dict[str, Any]:
    return {
        "model": model.path,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "dtype": args.dtype,
        "enable_sleep_mode": True,
        "enforce_eager": args.enforce_eager,
    }


def main(argv: list[str] | None = None) -> int:
    values = list(argv) if argv is not None else sys.argv[1:]
    args = parse_args(values)
    normalize_invocation_paths(args)
    reexec_with_python(
        args.python,
        "llm_switch_bench.experiments.backup_reuse_reclaim.run",
        values,
        workdir=args.vllm_repo,
        import_root=args.vllm_repo,
    )
    prepend_python_bin_to_path()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["VLLM_USE_V1"] = "1"

    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"vllm_repeated_sleep_l1_{time.time_ns()}"
    profile_path = out_dir / f"{run_id}.sleep_profile.jsonl"
    os.environ["VLLM_SLEEP_PROFILE_PATH"] = str(profile_path.resolve())
    if args.coordinator_url:
        os.environ["VLLM_CPU_BACKUP_COORDINATOR"] = "daemon"
        os.environ["VLLM_CPU_BACKUP_COORDINATOR_URL"] = args.coordinator_url
        os.environ["VLLM_CPU_BACKUP_COORDINATOR_TIMEOUT_S"] = str(args.coordinator_timeout_s)
    else:
        os.environ.pop("VLLM_CPU_BACKUP_COORDINATOR", None)
        os.environ.pop("VLLM_CPU_BACKUP_COORDINATOR_URL", None)
        os.environ.pop("VLLM_CPU_BACKUP_COORDINATOR_TIMEOUT_S", None)

    vllm_repo = module_repo_metadata("vllm")
    imported_vllm = None if vllm_repo is None else vllm_repo.get("module_path")
    expected_vllm = args.vllm_repo.resolve(strict=True)
    if imported_vllm is None or not Path(str(imported_vllm)).is_relative_to(expected_vllm):
        raise RuntimeError(
            f"imported vLLM from {imported_vllm}, outside --vllm-repo {expected_vllm}"
        )

    coordinator_provenance = None
    if args.coordinator_url:
        coordinator_repo = args.coordinator_repo.resolve(strict=True)
        coordinator_config = args.coordinator_config.resolve(strict=True)
        coordinator_provenance = {
            "repo": git_metadata(coordinator_repo),
            "config": file_metadata(coordinator_config),
        }

    summary: dict[str, Any] = {
        "run_id": run_id,
        "models": [{"name": model.name, "path": model.path} for model in args.models],
        "parameters": {
            "iterations": args.iterations,
            "cuda_visible_devices": args.cuda_visible_devices,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "dtype": args.dtype,
            "enforce_eager": args.enforce_eager,
            "prompt": args.prompt,
            "coordinator_url": args.coordinator_url,
            "coordinator_repo": (
                str(args.coordinator_repo.resolve()) if args.coordinator_repo else None
            ),
            "coordinator_config": (
                str(args.coordinator_config.resolve()) if args.coordinator_config else None
            ),
            "coordinator_timeout_s": args.coordinator_timeout_s,
            "post_release_observation_s": args.post_release_observation_s,
            "expect_release": args.expect_release,
            "expect_reuse": args.expect_reuse,
            "min_worker_rss_reclaim_bytes": args.min_worker_rss_reclaim_bytes,
        },
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "python_bin": str(Path(sys.executable).expanduser().absolute().parent),
            "ninja": command_output(["which", "ninja"]),
            "platform": platform.platform(),
            "benchmark_repo": git_metadata(repository_root()),
            "vllm_repo": vllm_repo,
            "coordinator": coordinator_provenance,
            "gpu": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ]
            ),
            "initial_meminfo_bytes": read_meminfo_bytes(),
        },
        "profile_path": str(profile_path),
        "out_dir": str(out_dir),
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
        last_release_bytes: dict[str, int] = {}
        reference_outputs: dict[str, tuple[list[int], str]] = {}
        profile_offset = 0
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
                            if args.coordinator_url:
                                record_coordinator_snapshot(
                                    step,
                                    "pre_wake",
                                    args.coordinator_url,
                                    run_id,
                                    args.coordinator_timeout_s,
                                )
                            activate_started = time.perf_counter()
                            llm.wake_up()
                            wake_ended = time.perf_counter()
                            activate_latency_s = wake_ended - activate_started
                            wake_events, profile_offset = load_profile_events_since(
                                profile_path, profile_offset
                            )
                            allocator_wake = newest_event(
                                events_in_window(
                                    wake_events,
                                    "allocator_wake_up",
                                    activate_started,
                                    wake_ended,
                                )
                            )
                            step.update(flatten_breakdown("wake_allocator", allocator_wake))
                            record_release_delta(
                                step,
                                "wake_allocator",
                                allocator_wake,
                                model.name,
                                last_release_bytes,
                            )
                            step["activate_type"] = "wake"
                            step["wake_latency_s"] = activate_latency_s
                            step["activate_latency_s"] = activate_latency_s
                            if args.post_release_observation_s > 0:
                                time.sleep(args.post_release_observation_s)
                            record_memory_snapshot(step, "post_wake", worker_pid)
                            if args.coordinator_url:
                                record_coordinator_snapshot(
                                    step,
                                    "post_wake",
                                    args.coordinator_url,
                                    run_id,
                                    args.coordinator_timeout_s,
                                )
                                step["wake_coordinator_release_delta_bytes"] = wake_reclaim_delta(
                                    step
                                )

                        llm = engines[model.name]
                        infer_started = time.perf_counter()
                        outputs = llm.generate(prompt, sampling_params, use_tqdm=False)
                        step["infer_latency_s"] = time.perf_counter() - infer_started
                        output = outputs[0].outputs[0]
                        token_ids = [int(token_id) for token_id in output.token_ids]
                        step["output_token_ids"] = token_ids
                        step["output_text"] = output.text
                        reference = reference_outputs.setdefault(
                            model.name, (token_ids, output.text)
                        )
                        step["output_matches_reference"] = reference == (
                            token_ids,
                            output.text,
                        )

                        worker_pid = engine_pids.get(model.name)
                        record_memory_snapshot(step, "pre_sleep", worker_pid)
                        sleep_started = time.perf_counter()
                        llm.sleep(level=1)
                        sleep_ended = time.perf_counter()
                        step["sleep_latency_s"] = sleep_ended - sleep_started
                        new_events, profile_offset = load_profile_events_since(
                            profile_path, profile_offset
                        )
                        allocator_sleep = newest_event(
                            events_in_window(
                                new_events,
                                "allocator_sleep",
                                sleep_started,
                                sleep_ended,
                            )
                        )
                        step.update(flatten_breakdown("sleep_allocator", allocator_sleep))
                        record_release_delta(
                            step,
                            "sleep_allocator",
                            allocator_sleep,
                            model.name,
                            last_release_bytes,
                        )
                        if allocator_sleep is not None and "pid" in allocator_sleep:
                            engine_pids[model.name] = int(allocator_sleep["pid"])
                        if args.post_release_observation_s > 0:
                            time.sleep(args.post_release_observation_s)
                        record_memory_snapshot(step, "post_sleep", engine_pids.get(model.name))
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

        coordinator_stats = None
        if args.coordinator_url:
            coordinator_stats = wait_for_run_coordinator_stats(
                args.coordinator_url,
                run_id,
                timeout_s=max(args.coordinator_timeout_s, 5.0),
                expect_release=args.expect_release is True,
            )
            summary["coordinator_stats"] = coordinator_stats
        assertion_failures = validate_results(args, steps, coordinator_stats)
        summary["assertion_failures"] = assertion_failures
        summary["ok"] = all(step.get("ok") for step in steps) and not assertion_failures
        summary["steps"] = steps
    except Exception as exc:
        summary["ok"] = False
        summary["error"] = repr(exc)
        summary["steps"] = steps
    finally:
        # Preserve partial evidence for failed runs as well as successful ones.
        # Ignore an incomplete trailing JSONL record still being written.
        evidence_errors: list[str] = []
        try:
            summary["sleep_profile_events"] = load_profile_events_since(profile_path, 0)[0]
        except Exception as exc:
            evidence_errors.append(f"profile: {exc!r}")
        try:
            summary["environment"]["final_meminfo_bytes"] = read_meminfo_bytes()
        except Exception as exc:
            evidence_errors.append(f"meminfo: {exc!r}")
        record_evidence_errors(summary, evidence_errors)
        summary_path = out_dir / "repeated_sleep_l1_summary.json"
        steps_csv_path = out_dir / "repeated_sleep_l1_steps.csv"
        summary["summary_path"] = str(summary_path)
        summary["steps_csv_path"] = str(steps_csv_path)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        write_steps_csv(steps_csv_path, steps)
        print(summary_path)
    return 0 if summary.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
