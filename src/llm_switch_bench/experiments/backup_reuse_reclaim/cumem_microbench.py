#!/usr/bin/env python3
"""Isolate vLLM CuMemAllocator sleep/wake copy bandwidth.

Creates synthetic CUDA tensors under CuMemAllocator's memory pool, then calls
allocator.sleep/offload and wake_up. This isolates whether the gap vs a plain
cudaMemcpy microbenchmark comes from VMM/cuMemMap allocations and/or many small
allocation copies rather than model execution.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import statistics
import time
from dataclasses import dataclass

import torch

from vllm.device_allocator.cumem import CuMemAllocator


@dataclass
class Row:
    case: str
    total_bytes: int
    chunks: int
    chunk_bytes_min: int
    chunk_bytes_max: int
    repeat: int
    sleep_latency_s: float
    wake_latency_s: float
    profile_copy_d2h_s: float
    profile_copy_h2d_s: float
    profile_create_map_s: float
    profile_unmap_release_s: float
    profile_cpu_backup_alloc_s: float
    d2h_gbps: float
    h2d_gbps: float


def parse_size(text: str) -> int:
    s = text.strip().lower()
    mult = 1
    if s.endswith("gib"):
        mult = 1024**3
        s = s[:-3]
    elif s.endswith("gb"):
        mult = 1000**3
        s = s[:-2]
    elif s.endswith("mib"):
        mult = 1024**2
        s = s[:-3]
    elif s.endswith("mb"):
        mult = 1000**2
        s = s[:-2]
    return int(float(s) * mult)


def split_even(total: int, chunks: int, alignment: int = 256 * 1024) -> list[int]:
    base = (total // chunks) // alignment * alignment
    sizes = [base] * chunks
    remain = total - base * chunks
    i = 0
    while remain >= alignment:
        sizes[i % chunks] += alignment
        remain -= alignment
        i += 1
    if remain:
        sizes[-1] += remain
    return [s for s in sizes if s > 0]


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


def run_case(
    case: str, total_bytes: int, chunks: int, repeats: int, out_dir: pathlib.Path
) -> list[Row]:
    rows: list[Row] = []
    sizes = split_even(total_bytes, chunks)
    for rep in range(repeats):
        profile_path = out_dir / f"{case}_rep{rep}.jsonl"
        if profile_path.exists():
            profile_path.unlink()
        os.environ["VLLM_SLEEP_PROFILE_PATH"] = str(profile_path)

        # Reset singleton to avoid old allocation metadata/pool state across cases.
        CuMemAllocator.instance = None
        allocator = CuMemAllocator.get_instance()
        tensors = []
        with allocator.use_memory_pool(tag="weights"):
            for idx, size in enumerate(sizes):
                t = torch.empty(size, dtype=torch.uint8, device="cuda")
                # Touch memory so the allocation is really resident/usable.
                t.fill_(idx % 251)
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

        # Verify contents survived wake.
        for idx, t in enumerate(tensors[: min(4, len(tensors))]):
            expected = idx % 251
            actual = int(t[0].item())
            if actual != expected:
                raise RuntimeError(
                    f"content mismatch case={case} rep={rep} idx={idx}: {actual} != {expected}"
                )

        sleep_ev, wake_ev = latest_events(profile_path)
        d2h = float(sleep_ev.get("copy_d2h_s", 0.0))
        h2d = float(wake_ev.get("copy_h2d_s", 0.0))
        rows.append(
            Row(
                case=case,
                total_bytes=sum(sizes),
                chunks=len(sizes),
                chunk_bytes_min=min(sizes),
                chunk_bytes_max=max(sizes),
                repeat=rep,
                sleep_latency_s=sleep_latency,
                wake_latency_s=wake_latency,
                profile_copy_d2h_s=d2h,
                profile_copy_h2d_s=h2d,
                profile_create_map_s=float(wake_ev.get("create_map_s", 0.0)),
                profile_unmap_release_s=float(sleep_ev.get("unmap_release_s", 0.0)),
                profile_cpu_backup_alloc_s=float(sleep_ev.get("cpu_backup_alloc_s", 0.0)),
                d2h_gbps=sum(sizes) / d2h / 1e9 if d2h else 0.0,
                h2d_gbps=sum(sizes) / h2d / 1e9 if h2d else 0.0,
            )
        )

        del tensors
        torch.cuda.empty_cache()
        allocator.cpu_backup_pool.clear()
        CuMemAllocator.instance = None
        if "VLLM_SLEEP_PROFILE_PATH" in os.environ:
            del os.environ["VLLM_SLEEP_PROFILE_PATH"]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--cases",
        default="1GiB:1,1GiB:41,3GiB:1,3GiB:76,6GiB:1,6GiB:104",
        help="comma-separated total:chunks cases",
    )
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[Row] = []
    for spec in args.cases.split(","):
        total_s, chunks_s = spec.split(":")
        total = parse_size(total_s)
        chunks = int(chunks_s)
        case = f"{total_s}_{chunks}chunks".replace(".", "p")
        print(f"running {case}", flush=True)
        all_rows.extend(run_case(case, total, chunks, args.repeats, out_dir))

    csv_path = out_dir / "cumem_copy_microbench.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(Row.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row.__dict__)

    print(f"wrote {csv_path}")
    grouped: dict[str, list[Row]] = {}
    for row in all_rows:
        grouped.setdefault(row.case, []).append(row)
    for case, rows in grouped.items():
        print(
            case,
            f"d2h_mean={statistics.mean(r.d2h_gbps for r in rows):.2f}GB/s",
            f"h2d_mean={statistics.mean(r.h2d_gbps for r in rows):.2f}GB/s",
            f"create_map_mean={statistics.mean(r.profile_create_map_s for r in rows):.4f}s",
        )


if __name__ == "__main__":
    main()
