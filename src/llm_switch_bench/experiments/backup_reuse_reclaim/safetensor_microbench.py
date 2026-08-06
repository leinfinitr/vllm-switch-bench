#!/usr/bin/env python3
"""CuMemAllocator copy benchmark using safetensors tensor byte sizes.

This approximates model-weight allocation granularity: allocate one CUDA uint8 tensor
per safetensors tensor byte size under vLLM's CuMemAllocator, then measure sleep
D2H and wake H2D profile fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import statistics
import struct
import time
from dataclasses import dataclass

import torch
from vllm.device_allocator.cumem import CuMemAllocator


@dataclass
class Row:
    model_dir: str
    repeat: int
    total_bytes: int
    chunks: int
    min_bytes: int
    median_bytes: int
    max_bytes: int
    sleep_latency_s: float
    wake_latency_s: float
    copy_d2h_s: float
    copy_h2d_s: float
    create_map_s: float
    unmap_release_s: float
    cpu_backup_alloc_s: float
    d2h_gbps: float
    h2d_gbps: float


def safetensor_sizes(model_dir: pathlib.Path) -> list[int]:
    sizes: list[int] = []
    for path in sorted(model_dir.glob("*.safetensors")):
        with path.open("rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        for key, val in header.items():
            if key == "__metadata__":
                continue
            start, end = val["data_offsets"]
            sizes.append(end - start)
    if not sizes:
        raise FileNotFoundError(f"no safetensors found in {model_dir}")
    return sizes


def latest_events(profile_path: pathlib.Path) -> tuple[dict, dict]:
    sleep = {}
    wake = {}
    for line in profile_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        if ev.get("phase") == "allocator_sleep":
            sleep = ev
        elif ev.get("phase") == "allocator_wake_up":
            wake = ev
    return sleep, wake


def bench_model(model_dir: pathlib.Path, repeats: int, out_dir: pathlib.Path) -> list[Row]:
    rows: list[Row] = []
    sizes = safetensor_sizes(model_dir)
    for rep in range(repeats):
        profile_path = out_dir / f"{model_dir.name}_rep{rep}.jsonl"
        os.environ["VLLM_SLEEP_PROFILE_PATH"] = str(profile_path)
        CuMemAllocator.instance = None
        allocator = CuMemAllocator.get_instance()
        tensors = []
        with allocator.use_memory_pool(tag="weights"):
            for idx, size in enumerate(sizes):
                t = torch.empty(size, dtype=torch.uint8, device="cuda")
                # Touch a byte to validate wake later without spending time filling all tensors.
                t[0] = idx % 251
                tensors.append(t)
        torch.cuda.synchronize()
        started = time.perf_counter()
        allocator.sleep(offload_tags=("weights",))
        torch.cuda.synchronize()
        sleep_latency = time.perf_counter() - started
        started = time.perf_counter()
        allocator.wake_up(tags=["weights"])
        torch.cuda.synchronize()
        wake_latency = time.perf_counter() - started
        for idx in (0, len(tensors) // 2, len(tensors) - 1):
            actual = int(tensors[idx][0].item())
            expected = idx % 251
            if actual != expected:
                raise RuntimeError(
                    f"mismatch {model_dir} rep={rep} idx={idx}: {actual} != {expected}"
                )
        sleep_ev, wake_ev = latest_events(profile_path)
        total = sum(sizes)
        d2h = float(sleep_ev.get("copy_d2h_s", 0.0))
        h2d = float(wake_ev.get("copy_h2d_s", 0.0))
        rows.append(
            Row(
                model_dir=str(model_dir),
                repeat=rep,
                total_bytes=total,
                chunks=len(sizes),
                min_bytes=min(sizes),
                median_bytes=int(statistics.median(sizes)),
                max_bytes=max(sizes),
                sleep_latency_s=sleep_latency,
                wake_latency_s=wake_latency,
                copy_d2h_s=d2h,
                copy_h2d_s=h2d,
                create_map_s=float(wake_ev.get("create_map_s", 0.0)),
                unmap_release_s=float(sleep_ev.get("unmap_release_s", 0.0)),
                cpu_backup_alloc_s=float(sleep_ev.get("cpu_backup_alloc_s", 0.0)),
                d2h_gbps=total / d2h / 1e9 if d2h else 0.0,
                h2d_gbps=total / h2d / 1e9 if h2d else 0.0,
            )
        )
        del tensors
        torch.cuda.empty_cache()
        CuMemAllocator.instance = None
        del os.environ["VLLM_SLEEP_PROFILE_PATH"]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("models", nargs="+")
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[Row] = []
    for model in args.models:
        print(f"running {model}", flush=True)
        rows.extend(bench_model(pathlib.Path(model), args.repeats, out_dir))
    csv_path = out_dir / "cumem_safetensor_sizes_microbench.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(Row.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
    print(f"wrote {csv_path}")
    for row in rows:
        print(
            row.model_dir,
            f"chunks={row.chunks}",
            f"D2H={row.d2h_gbps:.2f}GB/s",
            f"H2D={row.h2d_gbps:.2f}GB/s",
            f"create_map={row.create_map_s:.4f}s",
        )


if __name__ == "__main__":
    main()
