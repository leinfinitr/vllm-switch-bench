#!/usr/bin/env python3
"""Run a model-agnostic command and capture exact-disk tier evidence."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_switch_bench.common.provenance import repository_root
from llm_switch_bench.common.resources import (
    process_tree_rss_bytes,
    read_meminfo_bytes,
)
from llm_switch_bench.experiments.exact_disk.evidence import (
    OUTPUT_FILE,
    PROFILE_FILE,
    RESOURCE_FILE,
    RUN_FILE,
    build_curated_artifacts,
    write_evidence_manifest,
)

ROOT = repository_root()
DEFAULT_BACKUP_ROOT = Path("runtime/exact-disk-backups")


def parse_model_spec(value: str) -> dict[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be explicit NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("model must be explicit NAME=PATH")
    return {"name": name, "path": path}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=parse_model_spec, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    parser.add_argument(
        "--allow-nonempty-backup-root",
        action="store_true",
        help=(
            "Allow pre-existing backup-root bytes. The default rejects them so "
            "disk-footprint growth is attributable to this run."
        ),
    )
    parser.add_argument("--sample-interval-s", type=float, default=0.25)
    parser.add_argument(
        "--worker-pid",
        type=int,
        default=None,
        help="External vLLM worker PID whose process tree owns pinned backups.",
    )
    parser.add_argument(
        "--profile-path",
        type=Path,
        default=None,
        help="Existing external worker JSONL path; copied into raw evidence after run.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help=(
            "Command to profile, after --. The runner exports profile, output "
            "observation, backup-root, and model environment variables."
        ),
    )
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a benchmark command is required after --")
    if args.sample_interval_s <= 0:
        parser.error("--sample-interval-s must be positive")
    if args.worker_pid is not None and (
        args.worker_pid <= 0 or not Path(f"/proc/{args.worker_pid}").exists()
    ):
        parser.error("--worker-pid must identify a live process")
    return args


def _git_metadata(path: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(path), *arguments],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = run("status", "--porcelain", "--untracked-files=all")
    return {
        "path": str(path.resolve()),
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "status_porcelain": status,
    }


def _optional_repo_metadata(env_name: str) -> dict[str, Any] | None:
    value = os.environ.get(env_name)
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    return _git_metadata(path) if path.is_dir() else {"path": str(path), "missing": True}


def _directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for candidate in path.rglob("*"):
        try:
            if candidate.is_file() and not candidate.is_symlink():
                total += candidate.stat().st_size
        except FileNotFoundError:
            # The producer may atomically publish/remove files during sampling.
            continue
    return total


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resource_record(
    backup_root: Path,
    worker_pid: int,
    started_monotonic_s: float,
    *,
    disk_footprint_bytes: int | None = None,
) -> dict[str, Any]:
    meminfo = read_meminfo_bytes()
    return {
        "schema_version": 1,
        "wall_time": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": time.monotonic() - started_monotonic_s,
        "worker_pid": worker_pid,
        "worker_rss_bytes": process_tree_rss_bytes(worker_pid),
        "mem_available_bytes": meminfo.get("MemAvailable"),
        "disk_footprint_bytes": (
            _directory_size_bytes(backup_root)
            if disk_footprint_bytes is None
            else disk_footprint_bytes
        ),
    }


def _sample_resources(
    path: Path,
    backup_root: Path,
    worker_pid: int,
    interval_s: float,
    stop_event: threading.Event,
    ready_event: threading.Event,
    started_monotonic_s: float,
    initial_disk_footprint_bytes: int,
) -> None:
    with path.open("w", encoding="utf-8", buffering=1) as handle:
        # Bind the footprint baseline to the pre-launch state even if a very
        # short command creates its bundle before this thread is scheduled.
        handle.write(
            json.dumps(
                _resource_record(
                    backup_root,
                    worker_pid,
                    started_monotonic_s,
                    disk_footprint_bytes=initial_disk_footprint_bytes,
                ),
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        ready_event.set()
        while not stop_event.wait(interval_s):
            handle.write(
                json.dumps(
                    _resource_record(backup_root, worker_pid, started_monotonic_s),
                    sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
        # Always retain a post-command footprint sample, including for commands
        # that finish before one regular sampling interval.
        handle.write(
            json.dumps(
                _resource_record(backup_root, worker_pid, started_monotonic_s),
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
    pgid = proc.pid
    if not _process_group_exists(pgid):
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and _process_group_exists(pgid):
        time.sleep(0.05)
    if _process_group_exists(pgid):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGKILL)
    if proc.poll() is None:
        proc.wait(timeout=10)


def _base_run_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evidence_tier": "local_raw",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "backup_root": str(args.backup_root.resolve()),
        "allow_nonempty_backup_root": args.allow_nonempty_backup_root,
        "sample_interval_s": args.sample_interval_s,
        "worker_pid": args.worker_pid,
        "command": args.command,
        "command_return_code": None,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "benchmark_repo": _git_metadata(ROOT),
            "vllm_repo": _optional_repo_metadata("LLM_SWITCH_BENCH_VLLM_REPO"),
            "controller_repo": _optional_repo_metadata("LLM_SWITCH_BENCH_CONTROLLER_REPO"),
            "producer_executable": str(Path(args.command[0]).expanduser()),
            "producer_cwd": str(ROOT),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.out_dir.resolve()
    if output_dir.exists():
        print(f"output directory already exists: {output_dir}", file=sys.stderr)
        return 2
    backup_root = args.backup_root.resolve()
    initial_disk_footprint_bytes = _directory_size_bytes(backup_root)
    if initial_disk_footprint_bytes and not args.allow_nonempty_backup_root:
        print(
            "backup root must be empty so footprint growth is run-local: "
            f"{backup_root} ({initial_disk_footprint_bytes} bytes)",
            file=sys.stderr,
        )
        return 2
    if args.dry_run:
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(parents=True)
        run_path = raw_dir / RUN_FILE
        _write_json(run_path, _base_run_metadata(args))
        print(output_dir)
        return 0

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True)
    run_path = raw_dir / RUN_FILE
    run_metadata = _base_run_metadata(args)
    _write_json(run_path, run_metadata)

    backup_root.mkdir(parents=True, exist_ok=True)
    profile_path = raw_dir / PROFILE_FILE
    producer_profile_path = (
        args.profile_path.resolve() if args.profile_path is not None else profile_path
    )
    if args.profile_path is not None:
        producer_profile_path.parent.mkdir(parents=True, exist_ok=True)
        producer_profile_path.write_text("", encoding="utf-8")
    output_observation_path = raw_dir / OUTPUT_FILE
    resource_path = raw_dir / RESOURCE_FILE
    stdout_path = raw_dir / "command.stdout.log"
    stderr_path = raw_dir / "command.stderr.log"
    environment = os.environ.copy()
    environment.update(
        {
            "VLLM_EXACT_DISK_BACKUP_ENABLED": "1",
            "VLLM_EXACT_DISK_BACKUP_DIR": str(backup_root),
            "VLLM_CPU_BACKUP_DISK_DIR": str(backup_root),
            "VLLM_SLEEP_PROFILE_PATH": str(producer_profile_path),
            "LLM_SWITCH_BENCH_OUTPUT_OBSERVATION": str(output_observation_path),
            "LLM_SWITCH_BENCH_MODEL_NAME": args.model["name"],
            "LLM_SWITCH_BENCH_MODEL_PATH": args.model["path"],
        }
    )

    started_monotonic_s = time.monotonic()
    proc: subprocess.Popen[str] | None = None
    sampler: threading.Thread | None = None
    stop_event = threading.Event()
    sampler_ready_event = threading.Event()
    return_code = 1
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        try:
            proc = subprocess.Popen(
                args.command,
                cwd=ROOT,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                text=True,
                start_new_session=True,
            )
            worker_pid = proc.pid if args.worker_pid is None else args.worker_pid
            sampler = threading.Thread(
                target=_sample_resources,
                args=(
                    resource_path,
                    backup_root,
                    worker_pid,
                    args.sample_interval_s,
                    stop_event,
                    sampler_ready_event,
                    started_monotonic_s,
                    initial_disk_footprint_bytes,
                ),
                name="exact-disk-resource-sampler",
                daemon=True,
            )
            sampler.start()
            if not sampler_ready_event.wait(timeout=5):
                raise RuntimeError("resource sampler did not initialize")
            return_code = proc.wait()
        except KeyboardInterrupt:
            return_code = 130
            if proc is not None:
                _terminate_process_group(proc)
        except OSError as exc:
            return_code = 127
            stderr.write(f"failed to launch command: {exc}\n")
        finally:
            stop_event.set()
            if sampler is not None:
                sampler.join(timeout=max(args.sample_interval_s * 2, 1.0))
            if proc is not None:
                _terminate_process_group(proc)

    observed_worker_pid = args.worker_pid
    if observed_worker_pid is None and proc is not None:
        observed_worker_pid = proc.pid
    run_metadata.update(
        command_return_code=return_code,
        worker_pid=observed_worker_pid,
        duration_s=time.monotonic() - started_monotonic_s,
        finished_at=datetime.now(timezone.utc).isoformat(),
    )
    _write_json(run_path, run_metadata)
    if producer_profile_path != profile_path and producer_profile_path.is_file():
        profile_path.write_bytes(producer_profile_path.read_bytes())

    # Failed commands remain raw-only. This prevents an invalid lifecycle sample
    # from being mistaken for a numeric baseline while preserving its blocker.
    evidence_files = [
        RUN_FILE,
        RESOURCE_FILE,
        "command.stdout.log",
        "command.stderr.log",
    ]
    for optional in (PROFILE_FILE, OUTPUT_FILE):
        if (raw_dir / optional).is_file():
            evidence_files.append(optional)
    write_evidence_manifest(raw_dir, evidence_files)
    if return_code != 0:
        print(output_dir)
        return return_code

    missing = [name for name in (PROFILE_FILE, OUTPUT_FILE) if not (raw_dir / name).is_file()]
    if missing:
        print(
            f"successful command did not produce required raw evidence: {missing}",
            file=sys.stderr,
        )
        return 2

    try:
        result = build_curated_artifacts(raw_dir, output_dir / "curated")
    except ValueError as exc:
        print(f"cannot curate exact-disk evidence: {exc}", file=sys.stderr)
        return 2
    print(output_dir)
    return 0 if result["assertions"]["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
