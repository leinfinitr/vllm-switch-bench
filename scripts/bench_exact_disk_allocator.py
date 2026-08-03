#!/usr/bin/env python3
"""Measure allocator exact-disk demotion and restore on a real CUDA device."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import torch
from vllm.device_allocator.cumem import CuMemAllocator


def mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bytes", type=int, default=1024 * 1024 * 1024)
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=Path("/home/ljl/research-systems/vllm-model-switch-controller/tmp"),
    )
    parser.add_argument("--direct-io", type=int, choices=(0, 1), default=1)
    parser.add_argument("--chunk-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.bytes <= 0 or args.bytes % 4096:
        parser.error("--bytes must be a positive multiple of 4096")

    args.backup_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="exact-disk-microbench-", dir=args.backup_root
    ) as backup_root:
        os.environ["VLLM_EXACT_DISK_BACKUP_ENABLED"] = "1"
        os.environ["VLLM_EXACT_DISK_BACKUP_DIR"] = backup_root
        os.environ["VLLM_EXACT_DISK_BACKUP_DIRECT_IO"] = str(args.direct_io)
        os.environ["VLLM_EXACT_DISK_BACKUP_CHUNK_BYTES"] = str(args.chunk_bytes)
        profile_path = Path(backup_root) / "profile.jsonl"
        os.environ["VLLM_SLEEP_PROFILE_PATH"] = str(profile_path)

        allocator = CuMemAllocator.get_instance()
        elements = args.bytes // 4
        with allocator.use_memory_pool(tag="weights"):
            tensor = torch.arange(elements, dtype=torch.float32, device="cuda")
        expected_sample = tensor[:: max(1, tensor.numel() // 4096)].clone()

        prepared = allocator.prepare_cpu_backup("weights")
        prepared_bytes = int(prepared["prepared_bytes"])
        spill = allocator.prepare_disk_backup("weights")
        allocator.prepare_sleep("weights")
        rss_before_release = int(
            Path("/proc/self/statm").read_text().split()[1]
        ) * os.sysconf("SC_PAGE_SIZE")
        mem_available_before_release = mem_available_bytes()
        with allocator.cpu_backup_lock:
            allocator.pending_cpu_backup_release_bytes += prepared_bytes
            released = allocator._drain_pending_cpu_backup_release_locked()
        allocator._flush_cpu_backup_host_cache()
        rss_after_release = int(
            Path("/proc/self/statm").read_text().split()[1]
        ) * os.sysconf("SC_PAGE_SIZE")
        mem_available_after_release = mem_available_bytes()

        allocator.sleep("weights", skip_prepare=True)
        restore_started = time.perf_counter()
        allocator.wake_up(["weights"])
        restore_wall_s = time.perf_counter() - restore_started
        equal = bool(
            torch.equal(
                tensor[:: max(1, tensor.numel() // 4096)],
                expected_sample,
            )
        )
        events = [
            json.loads(line)
            for line in profile_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        disk_restore = next(
            event for event in events if event.get("phase") == "exact_disk_restore"
        )
        result = {
            "schema_version": 1,
            "requested_tensor_bytes": args.bytes,
            "allocator_runtime_bytes": prepared_bytes,
            "disk_spill_bytes": int(spill["disk_backup_written_bytes"]),
            "disk_spill_s": float(spill["disk_backup_write_s"]),
            "disk_read_bytes": int(disk_restore["disk_read_bytes"]),
            "disk_read_s": float(disk_restore["disk_read_s"]),
            "disk_copy_h2d_s": float(disk_restore["disk_copy_h2d_s"]),
            "restore_wall_s": restore_wall_s,
            "released": released,
            "rss_before_release_bytes": rss_before_release,
            "rss_after_release_bytes": rss_after_release,
            "rss_delta_bytes": rss_after_release - rss_before_release,
            "mem_available_before_release_bytes": mem_available_before_release,
            "mem_available_after_release_bytes": mem_available_after_release,
            "mem_available_delta_bytes": (
                mem_available_after_release - mem_available_before_release
            ),
            "output_equal": equal,
            "direct_io": bool(args.direct_io),
            "chunk_bytes": args.chunk_bytes,
        }
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(result, sort_keys=True))
        allocator.cleanup_exact_disk_backups()
        if not equal or not released:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
